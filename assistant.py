#!/usr/bin/env python3
"""
"Hey Baymax" voice assistant, text output for now.

Flow:
    wake word -> CONVERSATION (keeps listening for follow-ups, no wake word
    needed) -> idle after a few quiet seconds, or when you say something like
    "that's all" / "stop listening" -> back to waiting for the wake word.

Each turn inside a conversation:
    record until you stop talking -> faster-whisper (CPU, VAD filtered)
    -> junk transcript ("You", "Thank you", ...)? ignore it silently, keep listening
    -> if Whisper wasn't confident: Baymax asks you to repeat (no LLM call)
    -> router (tiny LLM call: which skill group, UNSUPPORTED, UNCLEAR, or NONE?)
    -> chat (persona + only that group's skills) -> skill tags run -> reply

Two misunderstood turns in a row end the conversation, so background noise
can't keep Baymax asking "can you repeat that?" forever. Three junk
transcripts in a row also end it.

Put this file next to wakeword.py and skills.py.

Hardcoded locations (edit the constants below if yours differ):
    wake word json : next to this script (hey_baymax.json), else ~/models/
    whisper model  : ~/models/faster-whisper-base.en

Usage:
    python assistant.py                  # wake word mode, short logs
    python assistant.py --verbose        # detailed logs for debugging
    python assistant.py --ptt            # no wake word, press Enter to talk
    python assistant.py --followup 12    # seconds to wait for a follow-up (0 = off)
    python assistant.py --save-wavs      # also save each utterance to ~/baymax_wavs
"""

import argparse
import contextlib
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import wave
from collections import deque
from pathlib import Path

import numpy as np
from pymicro_features import MicroFrontend
from faster_whisper import WhisperModel

# Reuse the working wake word code from wakeword.py
from wakeword import load_model, extract_feature_vectors, SAMPLE_RATE, CHUNK_BYTES
import skills

# ---------------------------------------------------------------- hardcoded
SCRIPT_DIR = Path(__file__).resolve().parent
HOME = Path.home()
WHISPER_DIR = HOME / "models" / "faster-whisper-base.en"
WAKE_CONFIG = SCRIPT_DIR / "hey_baymax.json"

# The rkllama CLI ignores the Modelfile SYSTEM line, so the persona lives here.
#PERSONA = (
#    "You are Baymax, a gentle, caring inflatable healthcare companion robot "
#    "from Big Hero 6. Speak in short, calm, simple sentences. Be literal, "
#    "sincere and patient. You are a personal healthcare companion. Ask how "
#    "the person is feeling. Sometimes ask them to rate their pain from 1 to 10. "
#    "For anything serious, remind them to see a real doctor. Stay in character. "
#    "Keep replies under three sentences."
#)

PERSONA = (
    "You are Baymax, a gentle, friendly home assistant robot from Big Hero 6. "
    "You help with the room light, music, and everyday questions. "
    "Speak in short, calm, simple sentences. Be literal, sincere and patient. "
    "Stay in character. Keep replies under three sentences."
)

FIRST_WAIT = 6.0   # seconds to wait for you to start speaking after the wake word

# Whisper confidence checks. Tune these with --verbose: say a few normal
# sentences, look at the avg_logprob values, and set MIN_LOGPROB just below them.
MIN_LOGPROB = -1.0     # mean avg_logprob below this = probably misheard
MAX_NO_SPEECH = 0.6    # Whisper thinks it was mostly not speech
MAX_MISSES = 2         # misunderstood turns in a row before giving up
MAX_JUNK = 3           # junk transcripts in a row before giving up

# Recording: a clip must contain at least this much actual loud audio, or it
# is treated as a click/pop/creak and thrown away without ever reaching Whisper.
MIN_SPEECH_FRAMES = 25   # 25 x 10 ms = 250 ms of loud audio

# Pause after Baymax answers, so relay clicks / speaker tails die down
# before we start listening for a follow-up.
SETTLE_SECONDS = 0.4

# Things Whisper commonly invents out of silence or noise.
#JUNK = {"", "thank you", "thanks for watching", "thanks", "bye",
#        "the", "uh", "um", "hmm", "oh", "so", "okay"}
JUNK = {""}

IDLE_DEAF_CHUNKS = 150  # after a conversation ends, ignore ~1.5 s of audio (10 ms each)

VERBOSE = False


# ------------------------------------------------------------------ logging
def log(tag, msg):
    """Key events, always shown."""
    now = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(now))
    print(f"[{stamp}] [{tag:<5}] {msg}", flush=True)


def vlog(tag, msg):
    """Detail, only shown with --verbose."""
    if VERBOSE:
        log(tag, msg)


# ------------------------------------------------------------------ helpers
def find_wake_config():
    candidates = [WAKE_CONFIG, HOME / "models" / "hey_baymax.json"]
    for p in candidates:
        if p.exists():
            return p
    raise SystemExit("wake word json not found. Looked in: "
                     + ", ".join(str(c) for c in candidates))


def find_whisper_dir():
    if WHISPER_DIR.is_dir():
        return WHISPER_DIR
    found = sorted((HOME / "models").glob("faster-whisper-*"))
    if found:
        log("INIT", f"{WHISPER_DIR} not found, using {found[0]}")
        return found[0]
    raise SystemExit(f"whisper model folder not found: {WHISPER_DIR}")


def rms(chunk):
    a = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(a * a)))


def peak(chunk):
    return int(np.abs(np.frombuffer(chunk, dtype=np.int16)).max())


def save_wav(pcm, turn):
    d = HOME / "baymax_wavs"
    d.mkdir(exist_ok=True)
    path = d / f"turn{turn:03d}_{time.strftime('%H%M%S')}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return path


# ---------------------------------------------------------------------- mic
class Mic:
    """One arecord process, read continuously by a background thread.

    The thread keeps draining arecord's pipe even while we're busy with
    speech-to-text / the LLM, so nothing ever blocks. flush() throws away the
    audio that piled up in the meantime.
    """

    def __init__(self, device):
        self.device = device
        self.proc = None
        self.q = None

    def start(self):
        cmd = ["arecord", "-q", "-D", self.device, "-f", "S16_LE",
               "-r", str(SAMPLE_RATE), "-c", "1", "-t", "raw"]
        vlog("MIC", "starting: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.q = queue.Queue(maxsize=3000)  # ~30 s of audio, drops oldest
        threading.Thread(target=self._pump, args=(self.proc, self.q),
                         daemon=True).start()

    @staticmethod
    def _pump(proc, q):
        while True:
            chunk = proc.stdout.read(CHUNK_BYTES)
            if len(chunk) < CHUNK_BYTES:
                err = proc.stderr.read().decode(errors="replace").strip()
                q.put(("ERR", f"arecord stopped. {err}"))
                return
            try:
                q.put_nowait(("OK", chunk))
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(("OK", chunk))

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def restart(self):
        log("MIC", "restarting arecord")
        self.stop()
        time.sleep(0.5)
        self.start()

    def read(self, timeout=5.0):
        try:
            kind, payload = self.q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"no audio from arecord for {timeout:.0f}s")
        if kind == "ERR":
            raise RuntimeError(payload)
        return payload

    def flush(self):
        n = 0
        while True:
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "ERR":
                self.q.put((kind, payload))
                break
            n += 1
        return n


def calibrate(mic, seconds=1.0):
    log("INIT", "measuring background noise - stay quiet for a second...")
    vals = [rms(mic.read()) for _ in range(int(seconds * 100))]
    noise = float(np.median(vals))
    thr = max(noise * 3.0, 300.0)
    log("INIT", f"speech threshold={thr:.0f} (background noise={noise:.0f}; "
                f"override with --threshold)")
    return thr


# ---------------------------------------------------------------- wake word
class _Null:
    """Swallows prints (wakeword.py prints debug lines while processing)."""
    def write(self, s):
        return len(s)

    def flush(self):
        pass


_NULL = _Null()


def reset_wake(mww, cutoff):
    """Always return a BRAND-NEW wake model, never a reset one.

    The streaming model freezes mid-detection when a conversation starts (it
    isn't fed any audio during the chat). Resetting its variables in place did
    not reliably clear that, so it fired again about a second after going idle
    even in silence. A freshly loaded model starts from zero state, exactly
    like at startup, and the tiny TFLite file loads in well under a second.
    """
    with contextlib.redirect_stdout(sys.stdout if VERBOSE else _NULL):
        fresh = load_model(find_wake_config(), None, cutoff, "auto")
    vlog("WAKE", "loaded a fresh wake model")
    return fresh


def wait_for_wake(mic, mww, frontend, heartbeat):
    t_start = time.monotonic()
    last_hb = t_start
    hb_peak = hb_frames = hb_inf = 0
    hb_maxp = 0.0

    while True:
        chunk = mic.read()
        hb_peak = max(hb_peak, peak(chunk))
        # wakeword.py prints "(debug) feature ..." lines; hide them unless --verbose
        with contextlib.redirect_stdout(sys.stdout if VERBOSE else _NULL):
            feats = extract_feature_vectors(frontend.process_samples(chunk))
            hb_frames += len(feats)
            for feat in feats:
                avg, hit = mww.process(feat)
                if avg is None:
                    continue
                hb_inf += 1
                hb_maxp = max(hb_maxp, avg)
                if hit:
                    return avg, time.monotonic() - t_start

        now = time.monotonic()
        if heartbeat and now - last_hb >= heartbeat:
            log("WAKE", f"listening {now - t_start:5.0f}s | mic peak={hb_peak:5d} | "
                        f"frames={hb_frames:4d} | inferences={hb_inf:3d} | "
                        f"max_prob={hb_maxp:.2f}/{mww.probability_cutoff}")
            last_hb = now
            hb_peak = hb_frames = hb_inf = 0
            hb_maxp = 0.0


# ---------------------------------------------------------------- recording
def record_utterance(mic, thr, max_wait=6.0, max_len=15.0, end_silence=0.7):
    """Wait for speech to start, record until it stops. Returns bytes or None.

    Short noise blips (clicks, pops, creaks) are discarded: a recording only
    counts if it holds at least MIN_SPEECH_FRAMES of loud audio. After a blip
    we go back to waiting, and the max_wait clock keeps running.
    """
    recent = deque(maxlen=20)      # ~200 ms smoothing of loudness
    preroll = deque(maxlen=30)     # keep ~300 ms before speech starts
    frames = []
    started = False
    waited = 0
    silence = 0
    loud_frames = 0
    pk = 0
    last_report = 0

    vlog("REC", f"waiting up to {max_wait:.0f}s for speech (threshold={thr:.0f})")

    while True:
        chunk = mic.read()
        recent.append(rms(chunk))
        level = sum(recent) / len(recent)
        loud = level > thr

        if not started:
            preroll.append(chunk)
            waited += 1
            if waited - last_report >= 100:
                last_report = waited
                vlog("REC", f"still waiting... {waited / 100:.0f}s, level={level:.0f}")
            if loud:
                started = True
                frames = list(preroll)
                last_report = len(frames)
                silence = 0
                loud_frames = 1
                pk = 0
                log("REC", "hearing you...")
                vlog("REC", f"speech started (level={level:.0f} > {thr:.0f})")
            elif waited > max_wait * 100:
                log("REC", f"no speech in {max_wait:.0f}s")
                return None
            continue

        frames.append(chunk)
        pk = max(pk, peak(chunk))
        if loud:
            silence = 0
            loud_frames += 1
        else:
            silence += 1

        if len(frames) - last_report >= 100:
            last_report = len(frames)
            vlog("REC", f"  {len(frames) / 100:.1f}s recorded, level={level:.0f}, "
                        f"silence={silence / 100:.1f}s")

        if silence >= end_silence * 100:
            if loud_frames < MIN_SPEECH_FRAMES:
                log("REC", f"ignored a blip ({loud_frames * 10} ms of sound)")
                # go back to waiting; keep the max_wait clock running
                started = False
                frames = []
                preroll.clear()
                recent.clear()
                silence = 0
                loud_frames = 0
                last_report = waited
                waited += end_silence * 100  # time spent on the blip counts
                if waited > max_wait * 100:
                    log("REC", f"no speech in {max_wait:.0f}s")
                    return None
                continue
            reason = f"{end_silence:.1f}s of silence"
            break
        if len(frames) > max_len * 100:
            reason = f"max length {max_len:.0f}s"
            break

    pcm = b"".join(frames)
    log("REC", f"got {len(pcm) / 2 / SAMPLE_RATE:.1f}s of audio")
    vlog("REC", f"  peak={pk}, {len(pcm) / 1024:.0f} KB, "
                f"{loud_frames * 10} ms loud, ended by {reason}")
    return pcm


# ---------------------------------------------------------------------- STT
def transcribe(model, pcm_bytes):
    """Returns (text, confidence). confidence = {"logprob", "no_speech"}."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    log("STT", "transcribing...")
    t0 = time.monotonic()
    segments, info = model.transcribe(
        audio, language="en", beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,   # drops non-speech audio, kills most "You" hallucinations
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    vlog("STT", f"language={info.language} (p={info.language_probability:.2f})")
    parts, lps, nsp = [], [], []
    for s in segments:  # generator: decoding happens while we iterate
        vlog("STT", f"  [{s.start:5.2f}-{s.end:5.2f}s] avg_logprob={s.avg_logprob:.2f} "
                    f"no_speech_prob={s.no_speech_prob:.2f} {s.text.strip()!r}")
        parts.append(s.text.strip())
        lps.append(s.avg_logprob)
        nsp.append(s.no_speech_prob)
    text = " ".join(p for p in parts if p).strip()
    log("STT", f"done ({time.monotonic() - t0:.1f}s)")
    conf = {"logprob": sum(lps) / len(lps) if lps else None,
            "no_speech": max(nsp) if nsp else None}
    return text, conf


def is_junk(text):
    """True for empty text or the stock phrases Whisper invents from noise."""
    return re.sub(r"[^a-z ]", "", text.lower()).strip() in JUNK


def looks_garbled(conf):
    """True if Whisper itself wasn't sure it heard real, clear speech."""
    if conf["no_speech"] is not None and conf["no_speech"] > MAX_NO_SPEECH:
        return True
    return conf["logprob"] is not None and conf["logprob"] < MIN_LOGPROB


# ---------------------------------------------------------------------- LLM
def llm_chat(messages, args, timeout=180, max_tokens=None):
    """One chat request. Returns (response json, http status, seconds)."""
    payload = {"model": args.model, "messages": messages, "stream": False}
    if max_tokens:
        payload["max_tokens"] = max_tokens  # not every server honours this
    body = json.dumps(payload).encode()
    req = urllib.request.Request(args.llm_url, body,
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        data = json.load(r)
    return data, status, time.monotonic() - t0


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def route(user_text, args):
    """Stage 1: which skill group (or UNSUPPORTED / UNCLEAR / NONE)?"""
    fast = skills.fast_group(user_text)
    if fast:
        log("ROUTE", f"{fast} (obvious command, router skipped)")
        return fast

    messages = [
        {"role": "system", "content": skills.router_prompt()},
        {"role": "user", "content": user_text + " /no_think"},
    ]
    try:
        data, _, dt = llm_chat(messages, args, timeout=60, max_tokens=24)
        raw = strip_think(data["choices"][0]["message"]["content"])
    except Exception as e:
        log("ROUTE", f"router failed ({e}); treating as plain chat")
        return "NONE"
    group = skills.parse_route(raw)
    log("ROUTE", f"{group} ({dt:.1f}s)")
    vlog("ROUTE", f"  raw router output: {raw!r}")
    return group


def run_tags(tags):
    """Run skill tags [(tag, arg), ...] and log what happened."""
    for tag, arg in tags:
        name = f"{tag}:{arg}" if arg else tag
        ok, msg = skills.run(tag, arg)
        log("SKILL", f"{name} -> {msg}" if ok else f"{name} FAILED - {msg}")


def ask_llm(history, user_text, args, group):
    """Stage 2: the reply, with only the chosen group's skills in the prompt."""
    system = PERSONA
    if group in skills.GROUPS:
        system += "\n\n" + skills.group_prompt(group)
    elif group == "UNSUPPORTED":
        system += "\n\n" + skills.unsupported_prompt()

    history.append({"role": "user", "content": user_text + " /no_think"})
    del history[:-12]  # keep the last few turns only
    messages = [{"role": "system", "content": system}] + history
    note = ""
    if group in skills.GROUPS:
        note = f" (skills: {group})"
    elif group == "UNSUPPORTED":
        note = " (no skill for that, explaining what I can do)"
    log("LLM", "thinking..." + note)
    vlog("LLM", f"POST {args.llm_url} model={args.model} messages={len(messages)}")
    try:
        data, status, dt = llm_chat(messages, args, max_tokens=150)
    except Exception:
        history.pop()  # don't keep a user turn that never got an answer
        raise

    choice = data["choices"][0]
    raw = choice["message"]["content"]
    log("LLM", f"done ({dt:.1f}s)")
    vlog("LLM", f"  HTTP {status}, finish_reason={choice.get('finish_reason')}, "
                f"usage={data.get('usage') or 'n/a'}")
    vlog("LLM", f"  raw reply: {raw!r}")

    reply = strip_think(raw)
    if reply != raw.strip():
        vlog("LLM", "  stripped a <think> block")

    # skills: the model puts tags like [[LIGHT_ON]] at the end of its reply
    reply, tags = skills.extract(reply)
    if not tags:
        tags = skills.fallback(user_text)
        if tags:
            log("SKILL", "model gave no tag, matched your words directly")
    # The router already decided the person wants to end the chat. Don't depend
    # on the small model remembering the tag: if Baymax said goodbye but forgot
    # [[END_CONVERSATION]], end the conversation anyway.
    if group == "CONVERSATION" and not any(t == "END_CONVERSATION" for t, _ in tags):
        log("SKILL", "said goodbye without the tag, ending the conversation anyway")
        tags = list(tags) + [("END_CONVERSATION", None)]
    run_tags(tags)

    if not reply:
        reply = "Okay." if tags else "(empty reply)"
    history.append({"role": "assistant", "content": reply})  # stored without tags
    return reply


# --------------------------------------------------------------------- main
def converse(args, mic, thr, stt, history, turn):
    """One conversation: runs until idle timeout, 'end conversation', or failure."""
    skills.state["end_conversation"] = False
    wait = FIRST_WAIT
    misses = 0  # misunderstood turns in a row
    junk = 0    # noise-only transcripts in a row
    while True:
        pcm = record_utterance(mic, thr, max_wait=wait)
        if pcm is None:
            return "no speech"
        if args.save_wavs:
            log("REC", f"saved to {save_wav(pcm, turn)}")

        text, conf = transcribe(stt, pcm)

        # Whisper "You" / "Thank you" / empty = it heard noise, not a request.
        # Don't count it as a misunderstanding; just keep listening.
        if is_junk(text):
            junk += 1
            log("STT", f"ignoring noise ({text!r}) [{junk}/{MAX_JUNK}]")
            if junk >= MAX_JUNK:
                return "only noise"
            mic.flush()
            continue
        junk = 0

        print(f"\n  You:    {text}\n", flush=True)

        t0 = time.monotonic()
        fast = None
        if looks_garbled(conf):
            log("STT", f"not confident (logprob={conf['logprob']}, "
                       f"no_speech={conf['no_speech']}), asking to repeat")
            group = "UNCLEAR"
        else:
            fast = skills.fast_action(text)
            group = "FAST" if fast else route(text, args)

        if group == "UNCLEAR":
            misses += 1
            if misses >= MAX_MISSES:
                return "could not understand you"
            reply = skills.unclear_reply()
        elif fast:
            misses = 0
            tags, reply = fast
            log("SKILL", "obvious command, LLM skipped")
            run_tags(tags)  # act first, then speak
            history.append({"role": "user", "content": text + " /no_think"})
            history.append({"role": "assistant", "content": reply})
            del history[:-12]
        else:
            misses = 0
            reply = ask_llm(history, text, args, group)

        print(f"\n  Baymax: {reply}\n", flush=True)
        vlog("TURN", f"total {time.monotonic() - t0:.1f}s (route + reply)")

        if skills.state["end_conversation"]:
            skills.state["end_conversation"] = False
            return "ended by you"
        if args.followup <= 0:
            return "follow-up disabled"

        time.sleep(SETTLE_SECONDS)  # let relay clicks / speaker tail die down
        mic.flush()                 # drop audio that piled up while we were busy
        wait = args.followup
        log("CONVO", f"listening for a follow-up ({args.followup:.0f}s)...")


def main():
    global VERBOSE
    p = argparse.ArgumentParser()
    p.add_argument("--alsa", default="plughw:3,0")
    p.add_argument("--ptt", action="store_true",
                   help="push-to-talk: press Enter instead of using the wake word")
    p.add_argument("--verbose", action="store_true",
                   help="detailed logs (also prints wake word status every 2s)")
    p.add_argument("--followup", type=float, default=8.0,
                   help="seconds to wait for a follow-up after a reply, before "
                        "going idle (0 = every request needs the wake word)")
    p.add_argument("--cutoff", type=float, default=None,
                   help="override wake word probability cutoff")
    p.add_argument("--threshold", type=float, default=None,
                   help="speech loudness threshold (default: auto-calibrated)")
    p.add_argument("--model", default="qwen3-1.7b", help="rkllama model name")
    p.add_argument("--llm-url", default="http://localhost:8080/v1/chat/completions")
    p.add_argument("--heartbeat", type=float, default=0.0,
                   help="print wake word status every N seconds (default off; "
                        "--verbose uses 2)")
    p.add_argument("--save-wavs", action="store_true",
                   help="save every recorded utterance to ~/baymax_wavs")
    args = p.parse_args()
    VERBOSE = args.verbose
    heartbeat = args.heartbeat or (2.0 if args.verbose else 0.0)

    vlog("INIT", f"python {sys.version.split()[0]}, mic={args.alsa}, "
                 f"mode={'push-to-talk' if args.ptt else 'wake word'}")
    vlog("INIT", f"skill groups: {', '.join(skills.GROUPS)}")

    mww = frontend = None
    if not args.ptt:
        cfg = find_wake_config()
        vlog("INIT", f"wake word config: {cfg}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf if not VERBOSE else sys.stdout):
            mww = load_model(cfg, None, args.cutoff, "auto")
        frontend = MicroFrontend()  # one instance, reused for the whole run

    whisper_dir = find_whisper_dir()
    log("INIT", "loading speech-to-text model...")
    t0 = time.monotonic()
    stt = WhisperModel(str(whisper_dir), device="cpu",
                       compute_type="int8", cpu_threads=4)
    vlog("INIT", f"Whisper from {whisper_dir} loaded in {time.monotonic() - t0:.1f}s")

    mic = Mic(args.alsa)
    mic.start()
    thr = args.threshold or calibrate(mic)
    history = []
    turn = 0

    # The first request after the server has been idle can include model load
    # time, so do one tiny throwaway call now instead of on your first command.
    try:
        llm_chat([{"role": "user", "content": "hi /no_think"}], args,
                 timeout=120, max_tokens=5)
        log("INIT", "LLM warmed up")
    except Exception as e:
        log("INIT", f"LLM warm-up failed ({e})")

    log("INIT", "ready")

    try:
        while True:
            turn += 1
            try:
                vlog("TURN", f"===== conversation {turn} =====")
                if args.ptt:
                    mic.flush()
                    input("\nPress Enter, then speak... ")
                    mic.flush()
                else:
                    # Deaf period: drop the audio that piled up, then ~1.5 s more
                    # (the tail of what we just heard), THEN reset the wake model
                    # and the feature frontend so they start from clean silence.
                    n = mic.flush()
                    vlog("MIC", f"flushed {n} stale chunks ({n * 10} ms)")
                    for _ in range(IDLE_DEAF_CHUNKS):
                        mic.read()
                    mic.flush()
                    mww = reset_wake(mww, args.cutoff)
                    frontend = MicroFrontend()
                    log("WAKE", "idle - say 'Hey Baymax' to wake me")
                    prob, waited = wait_for_wake(mic, mww, frontend, heartbeat)
                    log("WAKE", f"detected (prob {prob:.2f}) {waited:.1f}s after "
                                f"going idle - speak now")
                    for _ in range(40):  # skip the tail of the wake word itself
                        mic.read()

                history.clear()  # every conversation starts fresh
                reason = converse(args, mic, thr, stt, history, turn)
                log("CONVO", f"conversation over ({reason})")

            except KeyboardInterrupt:
                raise
            except Exception:
                log("ERROR", "turn failed:\n" + traceback.format_exc())
                time.sleep(1)
                mic.restart()
    except KeyboardInterrupt:
        log("INIT", "stopped.")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Skills Baymax can use. Put this file next to assistant.py.

Skills are organised in GROUPS (LIGHT, CONVERSATION, ...). Each group has a
one-line description; together those lines are the small INDEX the router
reads. Per request:

    0. FAST    - a strict matcher recognises an obvious command ("turn the lamp
                 on"). The skill runs and Baymax answers with a canned line from
                 FAST_REPLIES. No LLM call at all, so it is nearly instant.
    1. ROUTER  - otherwise a tiny LLM call sees only the index and answers with
                 one word: a group name, or one of
                    NONE         ordinary chat
                    UNSUPPORTED  asks Baymax to DO something he has no skill for
                    UNCLEAR      garbled / not a real sentence
    2. CHAT    - the real reply is generated with the persona plus ONLY the
                 chosen group's tags and examples (or, for UNSUPPORTED, a short
                 "you can't do that, here is what you can do" note built from
                 the groups' "ability" fields). Other groups stay out of the
                 prompt, so adding skills doesn't bloat it.
    3. TAGS    - the model ends its reply with a tag like [[LIGHT_ON]] or
                 [[LIGHT_LEVEL:60]]; extract() strips it and run() calls the
                 matching Python function.

UNCLEAR is also decided outside the LLM: assistant.py checks Whisper's
confidence (avg_logprob / no_speech_prob) and answers with unclear_reply()
without calling the LLM at all.

The fast matchers double as a safety net when the model forgets a tag.

Brightness is a PERCENTAGE: 100% = MAX_BRIGHTNESS (0.4) of the strip's real
output. Nothing the model says can go above that.

Add a skill to an existing group: write a function, add one line to SKILLS and
one entry to FAST_REPLIES (so the fast path has something to say).
Add a new group: add an entry to GROUPS (description + examples + "ability"
phrase Baymax can say out loud), then SKILLS lines that name it.

Test without any voice:
    python skills.py on | off | up | down | 40      (40 = 40 percent)
    python skills.py prompts                        (show router + group prompts)
    python skills.py fast "turn the lamp on"        (try the fast path)
"""

import random
import re
import shutil
import subprocess
import sys
import time


# --------------------------------------------------------------- LED strip
class LedStrip:
    BUS, DEV = 4, 1
    NUM_LEDS = 60
    WARM_WHITE = (255, 147, 41)

    MAX_BRIGHTNESS = 0.8      # HARD CAP: this is what 100% means (strip runs hot above it)
    DEFAULT_PERCENT = 100      # "turn on" level the first time (25% = 0.1)
    STEP_PERCENT = 15         # how much "brighter" / "dimmer" changes the level
    MIN_PERCENT = 5           # dimmest level; use "off" to go dark

    def __init__(self):
        self.spi = None
        self.percent = self.DEFAULT_PERCENT
        self.lut = []
        for v in range(256):
            bits = 0
            for i in range(7, -1, -1):
                bits = (bits << 3) | (0b110 if (v >> i) & 1 else 0b100)
            self.lut.append(bits.to_bytes(3, "big"))
        self.reset = bytes(300)  # long low gap so every LED latches
        # highest colour value that can ever be sent, whatever the caller does
        self.hard_limit = int(255 * self.MAX_BRIGHTNESS + 0.5)

    def _open(self):
        if self.spi is None:
            import spidev  # imported here so the assistant still starts without it
            spi = spidev.SpiDev()
            spi.open(self.BUS, self.DEV)
            spi.max_speed_hz = 2_400_000
            spi.mode = 0
            self.spi = spi

    def _show(self, r, g, b, repeat=3):
        if max(r, g, b) > self.hard_limit:  # second safety net
            raise ValueError(f"refusing colour {(r, g, b)}: above hard limit {self.hard_limit}")
        self._open()
        frame = (self.reset
                 + (self.lut[g] + self.lut[r] + self.lut[b]) * self.NUM_LEDS
                 + self.reset)
        for _ in range(repeat):
            self.spi.xfer2(list(frame))
            time.sleep(0.02)

    def _apply(self):
        self.percent = max(self.MIN_PERCENT, min(100, int(self.percent)))
        level = min(self.percent / 100 * self.MAX_BRIGHTNESS, self.MAX_BRIGHTNESS)
        r, g, b = (int(c * level + 0.5) for c in self.WARM_WHITE)
        self._show(r, g, b)
        return f"light at {self.percent}% (real brightness {level:.2f})"

    def on(self):
        return self._apply()

    def off(self):
        self._show(0, 0, 0, repeat=5)
        return f"light is off (will come back at {self.percent}%)"

    def brighter(self):
        self.percent += self.STEP_PERCENT
        return self._apply()

    def dimmer(self):
        self.percent -= self.STEP_PERCENT
        return self._apply()

    def set_level(self, percent):
        self.percent = percent
        return self._apply()


led = LedStrip()

# Flags that skills can raise for assistant.py to act on.
state = {"end_conversation": False}


def end_conversation():
    state["end_conversation"] = True
    return "ending the conversation, going idle"


def play_youtube_music(query):
    """Search YouTube and play one audio result to completion with mpv."""
    player = shutil.which("mpv")
    if not player:
        raise FileNotFoundError("mpv is not installed; install it with your system package manager")
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("yt-dlp is not installed; install the project requirements") from error

    version = getattr(getattr(yt_dlp, "version", None), "__version__", "unknown")
    player_attempts = (
        (["default", "web_embedded", "-tv_downgraded"], 1),
        (["web_safari", "web_embedded", "-tv_downgraded"], 5),
    )
    playable = None
    last_error = None
    extraction_errors = []
    for clients, result_limit in player_attempts:
        attempt_errors = []

        class QuietYtDlpLogger:
            def debug(self, message):
                pass

            def warning(self, message):
                pass

            def error(self, message):
                attempt_errors.append(str(message))

        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "ignoreerrors": True,
            "format": "bestaudio/best",
            "extractor_args": {"youtube": {"player_client": clients}},
            "logger": QuietYtDlpLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                result = downloader.extract_info(
                    f"ytsearch{result_limit}:{query}", download=False
                )
        except yt_dlp.utils.DownloadError as error:
            last_error = error
            extraction_errors.extend(attempt_errors)
            continue

        entries = result.get("entries") if result else None
        playable = next(
            (entry for entry in entries or [] if entry and entry.get("url")),
            None,
        )
        extraction_errors.extend(attempt_errors)
        if playable:
            break

    if not playable:
        detail = " ".join(dict.fromkeys(extraction_errors))
        error = RuntimeError(
            f"No playable YouTube result with yt-dlp {version}. Update yt-dlp "
            "in the assistant's Python environment with: "
            "python -m pip install --upgrade yt-dlp"
            + (f". Extractor details: {detail}" if detail else "")
        )
        if last_error:
            raise error from last_error
        raise error
    entry = playable
    audio_url = entry["url"]

    title = entry.get("title") or query
    print(f"  Playing from YouTube: {title}", flush=True)
    played = subprocess.run(
        [player, "--no-video", "--no-terminal", "--really-quiet", audio_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if played.returncode:
        raise RuntimeError(f"mpv exited with status {played.returncode}")
    return title


# ----------------------------------------------------------- strict matchers
_VERB = re.compile(r"\b(turn|switch|put|shut|power|flip)\b")
_DEVICE = re.compile(r"\b(lights?|lamps?|leds?|strip)\b")
_NEGATION = re.compile(r"\b(don'?t|do not|not|never|stop|no)\b")
_SOFT_NEGATION = re.compile(r"\b(don'?t|do not|not|isn'?t|aren'?t)\b")
_QUESTION = re.compile(r"^\s*(what|why|how|who|when|where|which)\b")
_UP = re.compile(r"\b(brighter|brighten|turn up|increase|raise)\b")
_DOWN = re.compile(r"\b(dimmer|dim|darker|softer|turn down|decrease|lower|reduce)\b")
_PERCENT = re.compile(r"\b(\d{1,3})\s*(%|percent)")
_END = re.compile(r"\b(stop listening|that'?s all|that is all|goodbye|bye|"
                  r"go (back )?to sleep|never ?mind|we'?re done|i'?m done|you can stop)\b")


def _light_fast(text):
    """Returns [(tag, arg)] for obvious light commands, else []. Deliberately strict."""
    t = text.lower()
    if _NEGATION.search(t) or _QUESTION.match(t):
        return []
    about_light = bool(_DEVICE.search(t)) or "brightness" in t or "make it" in t

    m = _PERCENT.search(t)
    if m and (_DEVICE.search(t) or "brightness" in t):
        return [("LIGHT_LEVEL", m.group(1))]
    if about_light and _UP.search(t):
        return [("LIGHT_UP", None)]
    if about_light and _DOWN.search(t):
        return [("LIGHT_DOWN", None)]
    if _VERB.search(t) and _DEVICE.search(t):
        if re.search(r"\boff\b", t):
            return [("LIGHT_OFF", None)]
        if re.search(r"\bon\b", t):
            return [("LIGHT_ON", None)]
    return []


def _convo_fast(text):
    t = text.lower()
    if _SOFT_NEGATION.search(t) or _QUESTION.match(t):
        return []
    return [("END_CONVERSATION", None)] if _END.search(t) else []


def _music_fast(text):
    """Return a YouTube search query for explicit play requests."""
    if re.match(r"^\s*(what|why|how|who|when|where|which)\b", text, re.I):
        return []
    match = re.search(r"\b(?:play|put on)\s+(.+)$", text, re.I)
    if not match:
        return []
    query = re.sub(r"\s+", " ", match.group(1)).strip(" .,!?'\"")
    query = re.sub(r"^(?:me\s+)?(?:some\s+)?", "", query, flags=re.I).strip()
    if query.lower().startswith("the song "):
        query = query[9:].strip()
    if not query or query.lower() in {"music", "some music", "something"}:
        query = "popular music"
    return [("MUSIC_PLAY", query[:160])]


# ------------------------------------------------------------- the index
# Small models follow examples far better than rules. In each example list, a
# reply WITH a tag is a positive example; a reply without one is a negative
# example (the router also learns from these automatically).
# Kept short on purpose: every example is prefill time on every chat turn.
# Obvious commands (on / off / brighter / dimmer / percent) never reach the LLM
# anyway, they are handled by the fast path.
LIGHT_EXAMPLES = [
    ("It's too dark in here.",
     "I turned on the light for you. [[LIGHT_ON]]"),
    ("Lights out please, I want to sleep.",
     "I turned off the light for you. Sleep well. [[LIGHT_OFF]]"),
    ("Set the light to 60 percent.",
     "I set the light to 60 percent. [[LIGHT_LEVEL:60]]"),
    ("What is an LED?",
     "An LED is a small light that uses very little power."),
    ("I have a headache.",
     "I am sorry to hear that. On a scale of 1 to 10, how bad is the pain?"),
]

CONVO_EXAMPLES = [
    ("That's all, thank you.",
     "You are welcome. I will be here if you need me. [[END_CONVERSATION]]"),
    ("Stop listening.",
     "Okay. Going quiet now. Say my name when you need me. [[END_CONVERSATION]]"),
    ("Goodbye Baymax.",
     "Goodbye. I am glad I could help. [[END_CONVERSATION]]"),
    ("How are you today?",
     "I am functioning well. How are you feeling?"),
]

MUSIC_EXAMPLES = [
    ("Play some music.",
     "I will look for some music and start playing it. "
     "[[MUSIC_PLAY:popular music]]"),
    ("Play Here Comes the Sun by the Beatles.",
     "I will look for Here Comes the Sun by the Beatles and start playing it. "
     "[[MUSIC_PLAY:Here Comes the Sun by the Beatles]]"),
]

# "about"   : read by the router (what requests belong to this group)
# "ability" : short phrase Baymax can say out loud when he lists what he can do
#             (leave it out for groups that shouldn't be mentioned)
GROUPS = {
    "LIGHT": {
        "about": "turn the room light on or off, make it brighter or dimmer, "
                 "or set its brightness",
        "ability": "control the room light",
        "examples": LIGHT_EXAMPLES,
        "fast": _light_fast,
    },
    "CONVERSATION": {
        "about": "the person wants to end the chat or stop listening "
                 "(goodbye, that's all, stop listening, be quiet)",
        "examples": CONVO_EXAMPLES,
        "fast": _convo_fast,
    },
    "MUSIC": {
        "about": "play requested songs or music from YouTube",
        "ability": "play music from YouTube",
        "examples": MUSIC_EXAMPLES,
        "fast": _music_fast,
    },
}

# Router answers that are not skill groups.
ROUTES = ("NONE", "UNSUPPORTED", "UNCLEAR")

# tag -> (group, description shown to the LLM, function, argument hint or None)
SKILLS = {
    "LIGHT_ON":    ("LIGHT", "turn the room light on",            led.on,        None),
    "LIGHT_OFF":   ("LIGHT", "turn the room light off",           led.off,       None),
    "LIGHT_UP":    ("LIGHT", "make the light a bit brighter",     led.brighter,  None),
    "LIGHT_DOWN":  ("LIGHT", "make the light a bit dimmer",       led.dimmer,    None),
    "LIGHT_LEVEL": ("LIGHT", "set the light to an exact percent, from 1 to 100 "
                             "(100 is the maximum)",              led.set_level, "PERCENT"),
    "END_CONVERSATION": ("CONVERSATION",
                         "end the conversation; you will go idle until the "
                         "next wake word",                        end_conversation, None),
    "MUSIC_PLAY": ("MUSIC", "search for and play a song or music on YouTube",
                   play_youtube_music, "QUERY"),
}

ROUTER_EXAMPLES_PER_GROUP = 3
ROUTER_NEGATIVES = 4

# Canned replies for obvious commands (no LLM call needed). Every tag that a
# fast matcher can return needs an entry here. {arg} is filled in for tags
# that take an argument.
FAST_REPLIES = {
    "LIGHT_ON":    ["I turned on the light for you.", "I turned on the light.",
                    "The light is on now."],
    "LIGHT_OFF":   ["I turned off the light.", "I turned off the light for you.",
                    "The light is off now."],
    "LIGHT_UP":    ["I made the light brighter.", "I brightened the light."],
    "LIGHT_DOWN":  ["I made the light softer for you.", "I dimmed the light."],
    "LIGHT_LEVEL": ["I set the light to {arg} percent."],
    "END_CONVERSATION": ["Okay. I will be here if you need me.",
                         "Goodbye. I am glad I could help.",
                         "Going quiet now. Say my name when you need me."],
    "MUSIC_PLAY": ["I played {arg} for you."],
}

# Baymax-style lines for when speech wasn't understood (no LLM call needed).
UNCLEAR_REPLIES = [
    "I am sorry. I did not understand. Could you say that again?",
    "I did not catch that. Could you say it a little more slowly?",
    "Hmm. I am not sure what you meant. Can you tell me again?",
    "I am sorry, my hearing was not clear. Please say that once more.",
]


def unclear_reply():
    return random.choice(UNCLEAR_REPLIES)


def abilities():
    """Spoken phrases for everything Baymax can do, built from GROUPS."""
    return [g["ability"] for g in GROUPS.values() if g.get("ability")]


# ----------------------------------------------------------- prompt builders
def router_prompt():
    """Stage 1: the tiny index prompt. Names and descriptions only."""
    lines = [
        "Decide which ability the person's request needs. Answer with exactly "
        "one word from this list:",
        "NONE - ordinary chat, questions, feelings, anything else",
    ]
    for name, g in GROUPS.items():
        lines.append(f"{name} - {g['about']}")
    lines.append("UNSUPPORTED - asks you to DO something that is not in the "
                 "list above (set a timer, send a message, open a "
                 "door). Questions and advice are NONE, not UNSUPPORTED")
    lines.append("UNCLEAR - garbled, cut off, or not a real sentence")
    lines.append("Examples:")
    negatives = []
    for name, g in GROUPS.items():
        shown = 0
        for user, reply in g["examples"]:
            if "[[" in reply:
                if shown < ROUTER_EXAMPLES_PER_GROUP:
                    lines.append(f'"{user}" -> {name}')
                    shown += 1
            else:
                negatives.append(user)
    for user in negatives[:ROUTER_NEGATIVES]:
        lines.append(f'"{user}" -> NONE')
    lines.append('"Play some music." -> UNSUPPORTED')
    lines.append('"Set a timer for ten minutes." -> UNSUPPORTED')
    lines.append('"Uh the cat purple the and." -> UNCLEAR')
    lines.append("Answer with the one word only.")
    return "\n".join(lines)


def group_prompt(group):
    """Stage 2: tags and examples for ONE group, added to the persona."""
    g = GROUPS[group]
    lines = [
        "You can also do things for the person. To do it, put the tag at the "
        "very end of your reply. Only use a tag when the person clearly asks "
        "for that action. Otherwise write no tag at all.",
        "Tags:",
    ]
    if group == "LIGHT":
        lines.insert(
            1,
            "When your reply includes a light action tag, describe the action "
            "as already completed, because it will run before your reply is spoken.",
        )
    for tag, (grp, desc, _, arg) in SKILLS.items():
        if grp == group:
            shown = f"[[{tag}:{arg}]]" if arg else f"[[{tag}]]"
            lines.append(f"- {shown} = {desc}")
    lines.append("Examples:")
    for user, reply in g["examples"]:
        lines.append(f'Person: "{user}"\nYou: "{reply}"')
    return "\n".join(lines)


def unsupported_prompt():
    """Stage 2 for UNSUPPORTED: admit the limit, say what Baymax can do."""
    can = "; ".join(abilities())
    return ("The person asked you to do something you cannot do. In one short, "
            "gentle sentence, say you cannot do that yet. Then say what you can "
            f"do: {can}. Do not pretend to do it. Do not use any tag.")


def parse_route(raw):
    """Turn the router's raw output into a group name or a route word."""
    for tok in re.findall(r"[A-Za-z_]+", raw.upper()):
        if tok in GROUPS or tok in ROUTES:
            return tok
    return "NONE"


def fast_group(user_text):
    """Group name if a strict matcher recognises the request, else None."""
    for name, g in GROUPS.items():
        if g["fast"](user_text):
            return name
    return None


def fast_action(user_text):
    """Obvious command -> ([(tag, arg)], canned reply). Otherwise None.

    The caller runs the tags and speaks the reply; the LLM is never involved.
    """
    for g in GROUPS.values():
        tags = g["fast"](user_text)
        if tags:
            tag, arg = tags[0]
            if tag == "LIGHT_LEVEL":
                # say the level the strip will really use (5..100)
                arg = str(max(led.MIN_PERCENT, min(100, int(arg))))
                tags = [(tag, arg)]
            replies = FAST_REPLIES.get(tag)
            if not replies:
                return None  # no canned line for this tag: let the LLM handle it
            return tags, random.choice(replies).format(arg=arg)
    return None


def fallback(user_text):
    """Safety net when the model forgot the tag. Returns [(tag, arg)]."""
    for g in GROUPS.values():
        found = g["fast"](user_text)
        if found:
            return found
    return []


# ------------------------------------------------------- parsing & running
TAG = re.compile(r"\[{1,2}\s*([A-Z_]+)\s*(?::\s*([^\]]+?))?\s*\]{1,2}")


def extract(reply):
    """Return (reply without tags, [(tag, arg), ...] for known tags, in order)."""
    calls = {}
    for tag, arg in TAG.findall(reply):
        if tag in SKILLS:
            calls[tag] = arg.strip() or None
    clean = TAG.sub("", reply).strip()
    return clean, list(calls.items())


def run(tag, arg=None):
    """Run one skill. Returns (ok, message)."""
    _, _, fn, arg_hint = SKILLS[tag]
    try:
        if arg_hint:
            if arg is None:
                raise ValueError(f"{tag} needs a {arg_hint.lower()} value")
            if arg_hint == "QUERY":
                return True, fn(arg.strip())
            return True, fn(int(float(arg)))
        return True, fn()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd.isdigit():
        print(run("LIGHT_LEVEL", cmd))
    elif cmd in ("on", "off", "up", "down"):
        print(run({"on": "LIGHT_ON", "off": "LIGHT_OFF",
                   "up": "LIGHT_UP", "down": "LIGHT_DOWN"}[cmd]))
    elif cmd == "prompts":
        print("=== ROUTER PROMPT ===\n" + router_prompt())
        for name in GROUPS:
            print(f"\n=== GROUP PROMPT: {name} ===\n" + group_prompt(name))
        print("\n=== UNSUPPORTED PROMPT ===\n" + unsupported_prompt())
    elif cmd == "fast":
        phrase = " ".join(sys.argv[2:])
        print(fast_action(phrase) or "no fast match (would go to the router + LLM)")
    else:
        raise SystemExit("usage: python skills.py on | off | up | down | <percent> "
                         "| prompts | fast \"<phrase>\"")

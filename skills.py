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

Add a skill to an existing group: write a function and add one line to SKILLS.
For a light command handled by the fast path, also add a FAST_REPLIES entry.
Add a new group: add an entry to GROUPS (description and, when appropriate, an
"ability" phrase Baymax can say out loud), then SKILLS lines that name it.

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


def search_youtube_music(query):
    """Return available song candidates from YouTube Music."""
    try:
        from ytmusicapi import YTMusic
    except ImportError as error:
        raise RuntimeError(
            "ytmusicapi is not installed; update requirements and install the project dependencies"
        ) from error

    matches = YTMusic().search(query, filter="songs", limit=20)
    candidates = []
    seen_ids = set()
    for match in matches:
        if (not isinstance(match, dict)
                or match.get("resultType") != "song"
                or match.get("isAvailable") is False):
            continue
        video_id = match.get("videoId")
        title = str(match.get("title") or "").strip()
        artists = match.get("artists") or []
        artist_names = [
            str(artist.get("name") or "").strip()
            for artist in artists if isinstance(artist, dict)
        ]
        if not video_id or not title or video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        album = match.get("album") or {}
        candidates.append({
            "video_id": video_id,
            "title": title,
            "artists": artist_names,
            "album": str(album.get("name") or "") if isinstance(album, dict) else "",
            "duration": str(match.get("duration") or ""),
        })
    return candidates


def play_youtube_music_result(candidate):
    """Resolve and play a specific YouTube Music search result."""
    player = shutil.which("mpv")
    if not player:
        raise FileNotFoundError("mpv is not installed; install it with your system package manager")
    try:
        import yt_dlp
    except ImportError as error:
        raise RuntimeError("yt-dlp is not installed; install the project requirements") from error

    if not candidate.get("video_id") or not candidate.get("title"):
        raise ValueError("YouTube Music result is missing its video ID or title")
    player_attempts = (
        ["default", "web_embedded", "-tv_downgraded"],
        ["web_safari", "web_embedded", "-tv_downgraded"],
    )
    extraction_errors = []

    class QuietYtDlpLogger:
        def debug(self, message):
            pass

        def warning(self, message):
            pass

        def error(self, message):
            extraction_errors.append(str(message))

    last_error = None
    selected = None
    music_url = f"https://music.youtube.com/watch?v={candidate['video_id']}"
    for clients in player_attempts:
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "bestaudio/best",
            "extractor_args": {"youtube": {"player_client": clients}},
            "logger": QuietYtDlpLogger(),
        }
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                entry = downloader.extract_info(music_url, download=False)
        except yt_dlp.utils.DownloadError as error:
            last_error = error
            continue
        if entry and entry.get("url"):
            selected = (candidate, entry["url"])
            break

    if not selected:
        details = " ".join(dict.fromkeys(extraction_errors))
        error = RuntimeError(
            "YouTube Music found matching songs, but yt-dlp could not resolve "
            "an audio stream."
            + (f" Extractor details: {details}" if details else "")
        )
        if last_error:
            raise error from last_error
        raise error

    candidate, audio_url = selected
    title = " - ".join(
        [candidate["title"], *[name for name in candidate["artists"] if name]]
    )
    print(f"  Playing from YouTube Music: {title} (Press 'C' to skip)", flush=True)
    proc = subprocess.Popen(
        [player, "--no-video", "--no-terminal", "--really-quiet", audio_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    import sys
    import select
    
    fd = sys.stdin.fileno()
    if sys.stdin.isatty():
        import tty
        import termios
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while proc.poll() is None:
                r, w, e = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    if ch.lower() == 'c':
                        print("  [Skipped song]", flush=True)
                        proc.terminate()
                        break
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    proc.wait()
    if proc.returncode and proc.returncode > 0:
        raise RuntimeError(f"mpv exited with status {proc.returncode}")
    return title


def play_youtube_music(query):
    """Compatibility wrapper that plays YouTube Music's first search result."""
    candidates = search_youtube_music(query)
    if not candidates:
        raise LookupError(f"YouTube Music returned no songs for {query!r}.")
    return play_youtube_music_result(candidates[0])


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


# ------------------------------------------------------------- the index
# Light examples help the model produce the correct tags. Other intents are
# classified from the request and recent conversation rather than fixed examples.
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
        "about": "the person wants to end the current conversation",
        "fast": _convo_fast,
    },
    "MUSIC": {
        "about": "the person wants music played; use their request and "
                 "conversation context to determine what they mean",
        "ability": "play music from YouTube Music",
        "examples": [],
    },
}

# Router answers that are not skill groups.
ROUTES = ("NONE", "WEB", "UNSUPPORTED", "UNCLEAR")

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
    "MUSIC_PLAY": ("MUSIC", "search YouTube Music for a song query",
                   play_youtube_music, "QUERY"),
}

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
    """Stage 1: classify intent from the request and recent conversation."""
    lines = [
        "Classify the person's current intent using the request and recent "
        "conversation. Interpret follow-ups in context, and choose the route "
        "that best matches what the person wants now. Do not rely on fixed "
        "phrases or assume a follow-up is unrelated. Answer with exactly one "
        "word from this list:",
    ]
    for name, g in GROUPS.items():
        lines.append(f"{name} - {g['about']}")
    lines.extend([
        "WEB - the person asks about real-world facts that may be specific, "
        "time-sensitive, or outside your reliable knowledge: charts, rankings, "
        "top or popular lists, best-of-year items, current events, release "
        "dates, award winners, statistics, lyrics, news, prices, or any factual "
        "claim you might get wrong. When in doubt, choose WEB.",
        "NONE - ordinary conversation, personal feelings, or subjective opinions "
        "that genuinely do not need outside information.",
    ])
    lines.extend([
        "UNSUPPORTED - the person wants an action that is not available.",
        "UNCLEAR - the request cannot be understood.",
        "Answer with the one word only.",
    ])
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
    elif group == "MUSIC":
        lines.insert(
            1,
            "Use the request and recent conversation to decide what music "
            "action is intended and, when playback is requested, form a useful "
            "YouTube Music search query. Put the query in plain text in the "
            "MUSIC_PLAY tag.",
        )
    for tag, (grp, desc, _, arg) in SKILLS.items():
        if grp == group:
            shown = f"[[{tag}:{arg}]]" if arg else f"[[{tag}]]"
            lines.append(f"- {shown} = {desc}")
    if group != "MUSIC" and g.get("examples"):
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
        matcher = g.get("fast")
        if matcher and matcher(user_text):
            return name
    return None


def fast_action(user_text):
    """Obvious command -> ([(tag, arg)], canned reply). Otherwise None.

    The caller runs the tags and speaks the reply; the LLM is never involved.
    """
    for g in GROUPS.values():
        matcher = g.get("fast")
        if not matcher:
            continue
        tags = matcher(user_text)
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
        matcher = g.get("fast")
        if not matcher:
            continue
        found = matcher(user_text)
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

# Baymax Voice Assistant

A local voice assistant for ARM devices like the Orange Pi 5. It combines a wake word detector, local speech recognition, and text-to-speech so the assistant can listen, understand, and respond without sending audio to external services.

## Features

- Wake-word detection with a local on-device model
- Speech-to-text using Faster Whisper
- Text-to-speech via Piper
- YouTube Music song search and audio playback using ytmusicapi, yt-dlp, and mpv
- Local-time answers, online dictionary definitions, and free web search
- Local model storage under `~/models`
- Works well on Ubuntu/Debian-based ARM systems

## Requirements

- Python 3.10+
- `venv` support
- `arecord` / ALSA for microphone input
- `aplay` / ALSA for assistant voice playback
- `mpv` for music playback (`sudo apt install mpv` on Ubuntu/Debian)
- A compatible ARM64 or x86_64 Linux environment

## Installation

```bash
cd /path/to/baymax_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the system audio player used for music:

```bash
sudo apt install mpv
```

## Download Piper

Download the latest ARM64 release and extract it somewhere convenient, such as the repo root or a local bin directory:

```bash
cd /path/to/baymax_assistant
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz
```

This gives you the `piper` binary in the extracted folder.

## Model files

Place the wake-word and Whisper files in `~/models` so the scripts can find
them automatically. By default, Piper uses the bundled `piper/piper`
executable and the `baymax.onnx` voice model beside `assistant.py`.

Typical files include:

- `~/models/hey_baymax.json`
- `~/models/faster-whisper-base.en`
- custom voice files such as `*.onnx`

Example:

```bash
mkdir -p ~/models
```

## Running the assistant

Start the wake-word voice assistant. Replies are generated with Piper and
played through the default ALSA output:

```bash
python assistant.py
```

On first startup, the assistant generates the fixed skill responses and short
wake-word and web-search acknowledgements, then stores them in `voice_cache/`.
Later starts reuse those WAV files. This cache contains only fixed replies;
generated conversational replies and variable brightness replies are temporary
and removed after playback. Delete `voice_cache/` to remove the prebuilt audio
and have it recreated next startup.
Recorded microphone audio is not saved unless `--save-wavs` is enabled; when
enabled, utterances are saved under `~/baymax_wavs`.

## Time, definitions, and web search

- Time questions use the Orange Pi's local clock and configured timezone.
- Word definitions use the free DictionaryAPI.dev endpoint.
- Factual questions try your local SearXNG first, then DuckDuckGo and
  Wikipedia as fallbacks. Baymax gives sourced results to the local LLM to
  summarize and includes the search engine in its spoken attribution.
- A lightweight LLM router also recognizes knowledge questions phrased without
  search keywords and sends those to web search, while keeping ordinary
  conversation on the chat path.

These lookups require an internet connection. The spoken question is sent to
your SearXNG instance first; only if it is unavailable or returns no usable
results will the query be sent to DuckDuckGo or Wikipedia. Do not ask Baymax to
search for private information. Search services can be unavailable or limit
requests, and web results may be incomplete. Baymax will say when a search
returns no useful results rather than making up an answer.

By default, Baymax uses `http://localhost:8082/search` and requests SearXNG's
JSON output. Set another endpoint if needed:

```bash
python assistant.py --searxng-url http://localhost:8082/search
```

Web answers are attributed aloud, for example, “According to Wikipedia, ...”
or “According to google, ...”, using the engine name in the selected result.

Verbose mode for troubleshooting:

```bash
python assistant.py --verbose
```

Press-to-talk mode without wake-word detection:

```bash
python assistant.py --ptt
```

Ask Baymax to play music or a named track, for example, “Play some music” or
“Play Here Comes the Sun by the Beatles.” Baymax searches the YouTube Music
song catalogue and plays a song result through `mpv`, rather than taking the
first general YouTube video result. Baymax says it will look for the requested
song before starting the search. `yt-dlp` resolves the selected YouTube Music
track's audio stream URL, so `mpv` does not need its own YouTube download
helper. Playback runs to
completion before Baymax listens for another request. Baymax cannot hear a
voice command to stop the track while it is playing. Playback errors are
reported after the player exits. This uses free, unofficial YouTube search and
playback, requires an internet connection, and may be affected by YouTube or
yt-dlp changes.
The LLM interprets music requests using the recent conversation, creates a
YouTube Music search query, and then receives the actual song results (including
titles, artists, albums, and durations) to choose the best match. Playback is
limited to one of those returned results; if the model cannot identify a
suitable match, Baymax does not start playback. For broad requests, the model
chooses a search query based on the person's stated preferences.

YouTube Music search uses the public, unauthenticated `ytmusicapi` interface;
install it with the project's requirements. Audio extraction still uses
yt-dlp. If extraction stops working, update yt-dlp using the same Python
environment that runs Baymax:

```bash
python -m pip install --upgrade yt-dlp
```

For stream extraction Baymax first tries the `default` and `web_embedded`
clients, then retries `web_safari` and `web_embedded` if needed. Both strategies
exclude the problematic `tv_downgraded` client. Errors from a failed strategy
stay quiet if the fallback succeeds. mpv's playback progress is also hidden.

The Python 3.10 deprecation warning is from yt-dlp and does not stop playback.
Upgrade to Python 3.11 or newer when convenient to remove the warning and stay
on a supported Python version.

To use a different Piper executable or voice model:

```bash
python assistant.py --piper /path/to/piper \
  --voice-model /path/to/my_voice.onnx
```

Set a different cache location with `--voice-cache /path/to/cache`.
Piper's `--length-scale` and `--sentence-silence` options adjust pace and
pauses; the defaults are slightly slower and more measured. This changes
cadence, not the identity of the voice model. Generated WAVs include a short
silent lead-in to help prevent the first word being clipped while audio
hardware wakes up. Before speech synthesis, emoji and Markdown asterisks are
removed so they are not spoken aloud.

## TTS example with a custom model

```bash
echo "Hello from my Orange Pi 5!" | ./piper --model /path/to/my_voice.onnx --output_file output.wav
```

You can also point Piper at a model from your `~/models` folder or a local custom voice file.

## Notes

- The repo expects `hey_baymax.json` beside the script or under `~/models`.
- The Whisper model is expected under `~/models/faster-whisper-base.en` unless you adjust the constants in the code.
- If the wake-word or Whisper model is missing, the scripts will exit with a clear error telling you where they looked.

## Quick start summary

```bash
cd /path/to/baymax_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p ~/models
# add model files to ~/models
python assistant.py
```

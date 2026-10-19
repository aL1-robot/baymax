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

## Run with Docker

Docker is supported on Linux hosts (including ARM64 Orange Pi boards). The
container installs the Python dependencies, ALSA tools, and `mpv`. It uses the
host's sound devices and host networking, so it can reach the local LLM and
search services using their existing `localhost` URLs.

First, put the Faster Whisper model in `~/models/faster-whisper-base.en` and
make sure the wake-word and voice models included in this repository are
present. Then configure optional API keys and your timezone:

```bash
mkdir -p ~/models
cp .env.example .env
# Edit .env: set TAVILY_API_KEY if desired and set TZ, for example TZ=Asia/Taipei
```

Build and start Baymax:

```bash
docker compose up --build
```

The Compose service mounts `~/models` read-only at `/root/models`, persists
generated speech in `./voice_cache`, and passes `/dev/snd` through for ALSA.
The default microphone device remains `plughw:3,0`; if your host uses another
device, update the Compose command, for example:

```yaml
command: ["python", "assistant.py", "--alsa", "default"]
```

Speech output uses ALSA's default device unless `ALSA_OUTPUT_DEVICE` is set in
`.env`. If Piper runs but there is no sound, find the host playback device with
`aplay -l` and set the matching ALSA device, for example:

```dotenv
ALSA_OUTPUT_DEVICE=plughw:0,0
```

Compose passes `/dev/spidev4.1` through for the light skill, matching the SPI
bus and device configured in `skills.py`. Enable that SPI device on the host
and adjust the mapping if the board uses a different device path. The
container does not request privileged mode. Audio and SPI access require the
host devices to exist and be accessible to Docker.

Stop the service with Ctrl+C or `docker compose down`. To run in push-to-talk
mode instead, add `"--ptt"` to the command array in `docker-compose.yml`.

## Download Piper

Download the latest ARM64 release and extract it somewhere convenient, such as the repo root or a local bin directory:

```bash
cd /path/to/baymax_assistant
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
tar -xzf piper_arm64.tar.gz
```

This gives you the `piper` binary in the extracted folder.

## Model files

Place the wake-word and Whisper files in `~/models` if they are not beside
`assistant.py`. By default, Baymax loads both included wake-word configs,
`hey_baymax.json` and `yow_baymax.json`, and their corresponding TFLite models.
Piper uses the bundled `piper/piper` executable and the `baymax.onnx` voice
model beside `assistant.py`.

Typical files include:

- `~/models/hey_baymax.json`
- `~/models/faster-whisper-base.en`
- custom voice files such as `*.onnx`

Example:

```bash
mkdir -p ~/models
```

Each wake phrase needs its own trained microWakeWord model and JSON config.
By default, Baymax listens for both included phrases. To select different
configs, pass `--wake-config` once per phrase:

```bash
python assistant.py \
  --wake-config ~/models/hey_baymax.json \
  --wake-config ~/models/yow_baymax.json
```

The configs can be stored elsewhere, but each config's `model` entry must
point to its TFLite file relative to that config. If either included config is
not beside the script, Baymax also looks for it in `~/models`.

## Running the assistant

Start the wake-word voice assistant. Replies are generated with Piper and
played through the default ALSA output:

```bash
python assistant.py
```

Ordinary chat replies use the LLM server's OpenAI-compatible streaming API.
Baymax starts Piper speech as each complete sentence arrives, while the model
continues generating the rest of the reply. Web searches, music requests, and
skill-action replies remain buffered until their results or actions are ready.

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
- Factual questions try your local OpenSERP service first, then Tavily,
  SearXNG, DuckDuckGo, and Wikipedia as fallbacks. Baymax gives sourced results
  to the local LLM to summarize and includes the search engine in its spoken
  attribution.
- A lightweight LLM router also recognizes knowledge questions phrased without
  search keywords and sends those to web search, while keeping ordinary
  conversation on the chat path. For web searches, the router turns the request
  and recent context into a concise query before searching.

These lookups require an internet connection. The LLM forms a focused search
query from the request and recent conversation, preserving relevant names and
constraints; Baymax still uses the original request and search results to form
its answer. Searches first use the local
OpenSERP service at `http://127.0.0.1:7000/mega/search`, trying Google, Bing,
querying all three in `balanced` mode, requesting extracted text from the top
two pages and up to five results. This lets Baymax compare results across
engines and use page content when available, rather than relying only on a
single engine's snippets. Brave is not currently a supported OpenSERP engine.
If OpenSERP is unavailable or returns no usable results, Baymax tries Tavily
when `TAVILY_API_KEY` is configured, then SearXNG, DuckDuckGo, and Wikipedia.
Do not ask Baymax to search for private information. Search services can be
unavailable or limit requests, and web results may be incomplete.

Baymax passes the returned titles, text, URLs, and engine names to the answer
model. With `--verbose`, it logs the query and the evidence text sent to that
model. The model is instructed to use only information supported by retrieved
result text, and to say “The search results do not say anything more about
that” when the text does not contain enough information to answer.

Copy `.env.example` to `.env` and set your Tavily key there:

```bash
cp .env.example .env
# Edit .env and set TAVILY_API_KEY
```

The real shell environment takes precedence over values in `.env`. Keep `.env`
private; it is excluded from version control.

OpenSERP's default endpoint can be changed with `--openserp-url`:

```bash
python assistant.py --openserp-url http://127.0.0.1:7000/mega/search
```

If OpenSERP is unavailable, Baymax uses `http://localhost:8082/search` and
requests SearXNG's JSON output. Set another SearXNG endpoint if needed:

```bash
python assistant.py --searxng-url http://localhost:8082/search
```

Web answers are attributed aloud, for example, “According to Wikipedia, ...”
or “According to google, ...”, using the engine name in the selected result.

Verbose mode for troubleshooting:

```bash
python assistant.py --verbose
```

For web lookups, verbose logs include the search query. When the LLM router
selects web search, they also include its brief search rationale. High-confidence
factual questions are forced through the router as well, so they get a focused
query instead of searching the request verbatim.

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
The LLM interprets music requests using the recent conversation and creates a
YouTube Music search query. Baymax selects the closest matching result from
YouTube Music's returned titles and artists in code, rather than relying on a
second model response to follow a machine-readable selection format. If the
LLM's stated song title and artist conflict with its action query, the stated
title and artist are used for the search. For broad requests, the model chooses
a search query based on the person's stated preferences.
Requests for music information, such as lists or recommendations, use web
search; YouTube Music search is for requests to play music.

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

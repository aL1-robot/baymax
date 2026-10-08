# Baymax Voice Assistant

A local voice assistant for ARM devices like the Orange Pi 5. It combines a wake word detector, local speech recognition, and text-to-speech so the assistant can listen, understand, and respond without sending audio to external services.

## Features

- Wake-word detection with a local on-device model
- Speech-to-text using Faster Whisper
- Text-to-speech via Piper
- Local model storage under `~/models`
- Works well on Ubuntu/Debian-based ARM systems

## Requirements

- Python 3.10+
- `venv` support
- `arecord` / ALSA for microphone input
- `aplay` / ALSA for assistant voice playback
- A compatible ARM64 or x86_64 Linux environment

## Installation

```bash
cd /path/to/baymax_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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
wake-word acknowledgements, then stores them in `voice_cache/`. Later starts
reuse those WAV files. This cache contains only fixed replies; generated
conversational replies and variable brightness replies are temporary and
removed after playback. Delete `voice_cache/` to remove the prebuilt audio and
have it recreated next startup.
Recorded microphone audio is not saved unless `--save-wavs` is enabled; when
enabled, utterances are saved under `~/baymax_wavs`.

Verbose mode for troubleshooting:

```bash
python assistant.py --verbose
```

Press-to-talk mode without wake-word detection:

```bash
python assistant.py --ptt
```

To use a different Piper executable or voice model:

```bash
python assistant.py --piper /path/to/piper \
  --voice-model /path/to/my_voice.onnx
```

Set a different cache location with `--voice-cache /path/to/cache`.

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

# Getting Started

python install -r requirements.txt

# Download piper latest ARM64 release archive

wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz

# Extract the archive
tar -xvf piper_arm64.tar.gz


**TTS with custom model**
echo "Hello from my Orange Pi 5!" | ./piper --model /path/to/my_voice.onnx --output_file output.wav

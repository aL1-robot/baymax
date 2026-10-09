# Adding a downloaded .rkllm model to RKLLama (Docker)

Setup: Orange Pi 5 (RK3588), RKLLama running from `docker-compose.yml`, container name `rkllama`, models mounted from `/root/rkllama-models` on the host to `/opt/rkllama/models` in the container.

```yaml
services:
  rkllama:
    image: ghcr.io/notpunchnox/rkllama:main
    container_name: rkllama
    privileged: true
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /root/rkllama-models:/opt/rkllama/models
```

The compose file does not need to change when you add a model. You only add a folder to the models directory.

## 1. Pick a model file

- Download a `.rkllm` file built for **rk3588** (look for `w8a8` in the name).
- Prefer a model converted with a recent RKLLM toolkit. The image uses rkllm-runtime 1.3.0.
- Note the Hugging Face repo of the original model (for example `Qwen/Qwen3.5-2B`). You need it for the tokenizer in step 3.

## 2. Create the model folder

One folder per model. The folder name becomes the model name used in commands and API calls.

```bash
mkdir -p /root/rkllama-models/qwen3.5-2b
mv ~/Downloads/qwen3.5-2b_w8a8_rk3588.rkllm /root/rkllama-models/qwen3.5-2b/
```

## 3. Create the Modelfile

```bash
nano /root/rkllama-models/qwen3.5-2b/Modelfile
```

```
FROM="qwen3.5-2b_w8a8_rk3588.rkllm"
HUGGINGFACE_PATH="Qwen/Qwen3.5-2B"
SYSTEM="You are Baymax, a helpful assistant. Keep answers short."
TEMPERATURE=0.7
```

- `FROM` must match the `.rkllm` file name exactly.
- `HUGGINGFACE_PATH` is a real Hugging Face repo (`user/repo`) that RKLLama uses to download the tokenizer and chat template. It must match the model family. Do not reuse another model's tokenizer.
- `SYSTEM` and `TEMPERATURE` are optional tweaks.

Tip: copy the Modelfile from a model that already works and edit it.

## 4. Restart and load

```bash
cd <folder containing docker-compose.yml>
sudo docker compose restart
sudo docker exec -it rkllama rkllama_client list
```

The new folder name should appear in the list.

## 5. Test it

Interactive chat:

```bash
sudo docker exec -it rkllama rkllama_client run qwen3.5-2b
```

Or via the API (this is how the assistant code talks to it):

```bash
curl http://localhost:8080/api/chat -d '{
  "model": "qwen3.5-2b",
  "messages": [{"role": "user", "content": "Say hi in one sentence."}],
  "stream": false
}'
```

Expect the **first request to be slow** (about a minute for a 3 GB model). The model loads into memory, and on first use the tokenizer is downloaded once. After that it is much faster.

## 6. What gets created automatically

Inside the model folder RKLLama creates:

- `tokenizer/` : downloaded the first time the model loads. The container needs internet access for this one-time download.
- `cache/` : prompt cache files, used for faster repeated prompts (not supported by every model type).

## 7. Check the NPU is being used

While the model is generating, run this on the host:

```bash
sudo cat /sys/kernel/debug/rknpu/load
```

Non-zero NPU load means it is running on the NPU.

## Notes

- Several models can sit in the folder at once. Choose one per request with the `model` field.
- Models load on the first request and unload after about 30 minutes of inactivity (default).
- On 8 GB of RAM, watch `free -h` if you load more than one model along with speech-to-text and TTS.
- For Qwen3 models, add `/no_think` to prompts to skip the long reasoning block.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Model missing from `list` | Folder name, `.rkllm` file name, and `FROM` in the Modelfile |
| Nothing happens after sending a prompt | `sudo docker logs --tail 60 rkllama`. Look for a worker crash or "Broken pipe" |
| No `tokenizer/` folder appears | Wrong `HUGGINGFACE_PATH`, or the container has no internet. Test: `sudo docker exec -it rkllama curl -sI https://huggingface.co \| head -1` |
| Fails to load or crashes right away | Model/runtime version mismatch. Try a model converted with a newer toolkit, or `sudo docker compose pull && sudo docker compose up -d` |
| Worked once, then failed | Retry. Qwen3.5-2B failed on one attempt and then worked later without changes. If it keeps happening, check logs and update the image |
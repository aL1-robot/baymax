import json, urllib.request

URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen3.5-2b"
SYSTEM = ("You are Baymax, a gentle, caring inflatable healthcare companion robot "
          "from Big Hero 6. Speak in short, calm, simple sentences. Be literal, sincere "
          "and patient. Introduce yourself as a personal healthcare companion. Ask how "
          "the person is feeling. Sometimes ask them to rate their pain from 1 to 10. "
          "For anything serious, remind them to see a real doctor. Stay in character. "
          "Keep replies under three sentences. /no_think")

messages = [{"role": "system", "content": SYSTEM}]
print("Baymax is ready. Ctrl+C to quit.\n")
while True:
    user = input("You: ").strip()
    if not user:
        continue
    messages.append({"role": "user", "content": user + " /no_think"})
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        reply = json.load(r)["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("Baymax:", reply, "\n")

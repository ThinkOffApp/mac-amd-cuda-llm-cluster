#!/usr/bin/env python3
"""One chat request to a local OpenAI-compatible server; prints decode tok/s and time to first token.
Usage: python3 ask.py PORT MODEL_NAME MAX_TOKENS"""
import json, sys, time, urllib.request

PROMPT = "Explain in plain words what a Thunderbolt eGPU dock does and name two limits of using one."
port, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": n,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time(); first = None; usage = None
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for c in d.get("choices", []):
            dl = c.get("delta", {})
            if (dl.get("content") or dl.get("reasoning_content")) and first is None:
                first = time.time()
end = time.time(); ct = usage["completion_tokens"] if usage else None
print(json.dumps({"completion_tokens": ct, "first_token_s": round(first - t0, 3) if first else None,
                  "decode_tok_s": round((ct - 1) / (end - first), 1) if ct and first else None}))

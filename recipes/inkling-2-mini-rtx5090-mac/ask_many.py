#!/usr/bin/env python3
"""STREAMS chat requests with different prompts, all sent at once; prints the total generation rate.
Usage: python3 ask_many.py PORT MODEL_NAME STREAMS [MAX_TOKENS]   (default 200 tokens per stream)
Total tok/s = all completion tokens / wall time from sending the first request to the last one finishing."""
import json, sys, threading, time, urllib.request

TOPICS = ["a Thunderbolt eGPU dock", "a solid-state drive", "a heat pump", "a lithium-ion battery", "a mesh Wi-Fi network",
          "a hash table", "a compiler", "a garbage collector", "a load balancer", "a transformer language model",
          "a refrigerator", "a jet engine", "a bicycle gearbox", "a mechanical watch", "a hydroelectric dam",
          "a vaccine", "a credit card network", "a search engine index", "a GPS receiver", "a noise-cancelling headset",
          "an electric car charger", "a 3D printer", "a sourdough starter", "a sailboat keel", "a nuclear reactor",
          "a database index", "a CPU cache", "an operating system scheduler", "a web browser", "a solar panel",
          "a wind turbine", "a microwave oven"]


def one(port, model, n, prompt, out, i):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    usage = None
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                usage = json.loads(line[6:]).get("usage") or usage
    out[i] = usage["completion_tokens"] if usage else 0


if __name__ == "__main__":
    port, model, streams = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    prompts = [f"Explain in plain words what {TOPICS[i % len(TOPICS)]} does and name two limits of using one."
               + (f" (request {i + 1})" if i >= len(TOPICS) else "") for i in range(streams)]
    out = [0] * streams
    threads = [threading.Thread(target=one, args=(port, model, n, p, out, i)) for i, p in enumerate(prompts)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    print(json.dumps({"streams": streams, "completion_tokens": sum(out), "wall_s": round(wall, 2),
                      "total_tok_s": round(sum(out) / wall, 1)}))

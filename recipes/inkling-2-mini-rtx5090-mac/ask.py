#!/usr/bin/env python3
"""One chat request to a local OpenAI-compatible server; prints decode tok/s and time to first token.
Usage: python3 ask.py PORT MODEL_NAME MAX_TOKENS [prose|code|copy]   (default prose)"""
import json, sys, time, urllib.request

_DOCK = ("The dock sits between the laptop and the graphics card. It carries power for the card from its own supply, "
         "and it carries data over a single cable. Inside the cable the signal is split into lanes, and each lane runs "
         "at a fixed rate, so the total rate depends on how many lanes the host offers. A laptop that offers two lanes "
         "at twenty gigabits each gives forty in total, while a newer one offers eighty. The card itself does not care "
         "about the cable; it only sees a slot with fewer lanes than a desktop board would give it. For work that stays "
         "on the card, such as running a model whose weights are already loaded, the narrow link matters little, "
         "because only small results travel back. For work that moves large buffers every frame, the link becomes the "
         "limit. Heat is handled by the card's own fans, and the supply must be sized for the peak draw of the card "
         "rather than its average. When the cable is pulled while the card is busy, the driver loses the device in the "
         "middle of a transfer, and the safest recovery is to switch the dock off and on again.")

PROMPTS = {
    "prose": "Explain in plain words what a Thunderbolt eGPU dock does and name two limits of using one.",
    "code": ("Write a Python function parse_csv(path) that reads a CSV file with a header row and returns a list of "
             "dicts, with type hints, a docstring and error handling for a missing file. Then write three pytest tests "
             "for it. Code only, no explanations."),
    # the document-reproduction prompt: the paragraph three times, each followed by one space
    "copy": ("Repeat the following text exactly, word for word, with no changes and no commentary:\n\n"
             + (_DOCK + " ") * 3),
}


def ask(port, model, n, prompt):
    """Send one streamed greedy request; return (completion_tokens, first_token_s, decode_tok_s)."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
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
    return (ct, round(first - t0, 3) if first else None,
            round((ct - 1) / (end - first), 1) if ct and first and end > first else None)


if __name__ == "__main__":
    port, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    which = sys.argv[4] if len(sys.argv) > 4 else "prose"
    if which not in PROMPTS:
        sys.exit(f"ask.py: prompt must be one of {', '.join(PROMPTS)}")
    ct, ttft, tps = ask(port, model, n, PROMPTS[which])
    print(json.dumps({"prompt": which, "completion_tokens": ct, "first_token_s": ttft, "decode_tok_s": tps}))

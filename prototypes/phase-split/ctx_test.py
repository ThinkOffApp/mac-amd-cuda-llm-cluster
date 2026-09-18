"""Is the 156.9 MB "fixed block" a property of the MODEL, or of `-c`?

It matters because the crossover formula now being proposed treats it as a model
constant. If it is really a slice of the allocated context, then every crossover
computed from it is only valid at -c 4096, and a deployment at -c 32768 would get
a different answer from the same script.

Two candidates:
    architecture   Qwen3-Next carries linear-attention / recurrent state, which is
                   fixed size regardless of sequence length. Would not move with -c.
    allocation     the slot save writes the whole allocated cache. Would scale with -c.

Same prompt length, two servers differing only in -c.
"""
import json, subprocess, sys, time

def post(base, path, payload, t="600"):
    o = subprocess.run(["curl","-sS","--max-time",t,"-X","POST",base+path,
                        "-H","Content-Type: application/json","-w","\n__S__:%{http_code}",
                        "-d",json.dumps(payload)], capture_output=True, text=True)
    if o.returncode: sys.exit(f"curl exit {o.returncode}: {o.stderr[:200]}")
    body,_,st = o.stdout.rpartition("\n__S__:")
    if not 200 <= int(st) < 300: sys.exit(f"{path}: HTTP {st}: {body[:300]}")
    return json.loads(body)

toks = json.load(open("/tmp/q_full.json"))["prompt"][:512]
for base, label in [("http://127.0.0.1:8091", "-c 4096 (already running)"),
                    ("http://127.0.0.1:8093", "-c 8192 (started for this test)")]:
    post(base, "/slots/0?action=erase", {})
    r = post(base, "/completion", {"prompt": toks, "n_predict": 1, "cache_prompt": False,
                                   "temperature": 0.0})
    s = post(base, "/slots/0?action=save", {"filename": "ctxtest.bin"})
    n = s.get("n_written") or s.get("n_read")
    print(f"{label:<32} 512 tok -> slot {n:>12,} B")

"""Control: how much does the SAME slot file vary between two restores?

Without this number the 0.67 logprob gap between the M5-produced and
Mini-produced caches is uninterpretable -- Metal reductions are not
bit-deterministic across runs either, so some of that gap may be this machine
arguing with itself rather than the two backends disagreeing.
"""
import json, subprocess, sys

BASE = "http://127.0.0.1:8091"
REQ = json.load(open("/tmp/q_full.json"))

def post(path, payload):
    o = subprocess.run(["curl", "-sS", "--max-time", "300", "-X", "POST", BASE + path,
                        "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
                       capture_output=True, text=True)
    if o.returncode: sys.exit(f"curl failed {path}: {o.stderr.strip()}")
    return json.loads(o.stdout)

def run(slotfile):
    post("/slots/0?action=restore", {"filename": slotfile})
    b = dict(REQ); b.update(n_predict=4, cache_prompt=True, temperature=0.0,
                            top_k=0, top_p=1.0, n_probs=10)
    c = post("/completion", b)
    return c.get("content"), (c.get("completion_probabilities") or [])

def gap(pa, pb):
    worst = 0.0
    for i in range(min(len(pa), len(pb))):
        da = {e["token"]: e["logprob"] for e in (pa[i].get("top_logprobs") or pa[i].get("probs") or [])}
        db = {e["token"]: e["logprob"] for e in (pb[i].get("top_logprobs") or pb[i].get("probs") or [])}
        for k in set(da) & set(db):
            worst = max(worst, abs(da[k] - db[k]))
    return worst

pairs = [("pd.bin", "pd.bin"), ("q27.bin", "q27.bin"), ("pd.bin", "q27.bin")]
print(f"{'comparison':<26} {'same text':<10} worst |dlogprob|")
for a, b in pairs:
    ca, pa = run(a)
    cb, pb = run(b)
    kind = "SAME FILE twice" if a == b else "M5-KV vs Mini-KV"
    print(f"{a+' vs '+b:<26} {str(ca == cb):<10} {gap(pa, pb):.6f}   <- {kind}")

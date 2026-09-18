"""Is the 0.143 ms a FIXED per-wait overhead, or latency that real work hides?

My claim was "24 drains per token = 3.4 ms of pure waiting". That only holds if
the cost is fixed per wait. If substantial GPU work absorbs it, the penalty is
smaller. Test: run N real matmuls, waiting after each, versus the same N under a
single wait. The difference divided by (N-1) is the true per-extra-wait cost.
"""
import json, statistics as st, time, torch
torch.set_num_threads(1); dev = "mps"
REPS = 60
# a matmul of roughly GPT-2 block scale so the work is realistic
A = torch.randn(768, 768, device=dev); x = torch.randn(1, 768, device=dev)
out = {}

def med(fn, reps=REPS):
    for _ in range(20): fn()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize(); t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts) * 1000

def n_ops_one_wait(n):
    def f():
        y = x
        for _ in range(n): y = y @ A
    return f

def n_ops_n_waits(n):
    def f():
        y = x
        for _ in range(n):
            y = y @ A
            torch.mps.synchronize()
    return f

for n in (1, 6, 12, 24):
    one = med(n_ops_one_wait(n)); many = med(n_ops_n_waits(n))
    out[f"n{n}"] = {"one_wait_ms": one, "n_waits_ms": many,
                    "extra_ms": many - one,
                    "per_extra_wait_ms": (many - one) / max(n - 1, 1)}
print(json.dumps(out, indent=1))

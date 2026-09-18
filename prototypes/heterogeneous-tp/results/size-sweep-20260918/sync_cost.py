"""Is the fixed per-copy cost data movement, or just synchronisation?

Apple Silicon has unified memory: a device->host copy of 3 KB should not cost
0.15 ms in data terms. If that time is actually the MPS synchronise, the lever
is the number of syncs, not the number or size of copies.
"""
import json, statistics as st, time, torch

dev = torch.device("mps")
torch.set_num_threads(1)
REPS = 200
x = torch.randn(768, device=dev)          # one GPT-2 collective payload, 3 KB
out = {}

# 1. synchronize() on an idle queue, alone
for _ in range(20): torch.mps.synchronize()
ts = []
for _ in range(REPS):
    t0 = time.perf_counter(); torch.mps.synchronize(); ts.append(time.perf_counter() - t0)
out["synchronize_alone_ms"] = st.median(ts) * 1000

# 2. the copy as the harness does it: sync, copy, sync
ts = []
for _ in range(REPS):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    _ = x.to("cpu").contiguous()
    torch.mps.synchronize()
    ts.append(time.perf_counter() - t0)
out["copy_with_syncs_ms"] = st.median(ts) * 1000

# 3. copies back to back with ONE sync at the end, cost amortised per copy
for n in (1, 8, 24):
    for _ in range(5):
        for _ in range(n): _ = x.to("cpu")
        torch.mps.synchronize()
    ts = []
    for _ in range(REPS // 4):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(n): _ = x.to("cpu")
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t0) / n)
    out[f"copy_batched_{n}_per_copy_ms"] = st.median(ts) * 1000

print(json.dumps(out, indent=1))

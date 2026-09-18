"""Device->host copy cost versus payload size, on one machine.

Settles a question two data points could not: is the cost per-call or per-byte,
and at what size does bandwidth start to matter? That decides whether a narrower
wire dtype (FP16) can help at the sizes GPT-2 actually moves.
"""
import json, statistics as st, time, torch

dev = torch.device("mps")
torch.set_num_threads(1)
REPS = 60
rows = []
for floats in (256, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304, 196608, 393216):
    x = torch.randn(floats, device=dev)
    for _ in range(10):                      # warm
        _ = x.to("cpu")
    ts = []
    for _ in range(REPS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = x.to("cpu").contiguous()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    med = st.median(ts)
    rows.append({"floats": floats, "bytes": floats * 4, "median_ms": med * 1000,
                 "MB_per_s": (floats * 4) / 1e6 / med})
print(json.dumps({"device": "mps", "reps": REPS, "rows": rows}, indent=1))

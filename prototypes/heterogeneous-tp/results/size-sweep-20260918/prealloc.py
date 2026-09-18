"""Is the fixed per-copy cost allocation, or the transfer itself?

`.to("cpu")` allocates a fresh destination tensor every call. The transport
makes 24 of those per token. A preallocated destination with copy_() removes the
allocation while moving the same bytes.
"""
import json, statistics as st, time, torch

dev = torch.device("mps"); torch.set_num_threads(1)
REPS = 300
src = torch.randn(768, device=dev)
dst = torch.empty(768)                      # preallocated host buffer
out = {}

def timeit(fn, reps=REPS):
    for _ in range(30): fn()
    ts = []
    for _ in range(reps):
        torch.mps.synchronize()
        t0 = time.perf_counter(); fn(); torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return st.median(ts) * 1000

out["to_cpu_allocating_ms"] = timeit(lambda: src.to("cpu").contiguous())
out["copy_into_prealloc_ms"] = timeit(lambda: dst.copy_(src))
out["copy_into_prealloc_nonblocking_ms"] = timeit(lambda: dst.copy_(src, non_blocking=True))
# and the reverse direction, which the transport also pays
src_h = torch.randn(768); dst_d = torch.empty(768, device=dev)
out["to_device_allocating_ms"] = timeit(lambda: src_h.to(dev))
out["to_device_prealloc_ms"] = timeit(lambda: dst_d.copy_(src_h))
print(json.dumps(out, indent=1))

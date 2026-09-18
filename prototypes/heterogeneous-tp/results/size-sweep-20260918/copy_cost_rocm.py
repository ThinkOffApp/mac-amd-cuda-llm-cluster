"""Same copy-cost probe, ROCm side. Tiny load; may run alongside a serving model."""
import json, statistics as st, time, torch
dev = torch.device("cuda"); torch.set_num_threads(1)
def sync(): torch.cuda.synchronize()
REPS = 300
out = {"torch": torch.__version__, "hip": torch.version.hip}
for _ in range(30): sync()
ts=[]
for _ in range(REPS):
    t0=time.perf_counter(); sync(); ts.append(time.perf_counter()-t0)
out["synchronize_alone_ms"] = st.median(ts)*1000
rows=[]
for floats in (256, 768, 3072, 12288, 49152, 196608):
    x = torch.randn(floats, device=dev)
    for _ in range(20): _ = x.to("cpu")
    ts=[]
    for _ in range(REPS):
        sync(); t0=time.perf_counter(); _ = x.to("cpu").contiguous(); sync()
        ts.append(time.perf_counter()-t0)
    rows.append({"bytes": floats*4, "median_ms": st.median(ts)*1000})
out["device_to_host"]=rows
src=torch.randn(768, device=dev); dst=torch.empty(768)
def timeit(fn):
    for _ in range(30): fn()
    ts=[]
    for _ in range(REPS):
        sync(); t0=time.perf_counter(); fn(); sync(); ts.append(time.perf_counter()-t0)
    return st.median(ts)*1000
out["to_cpu_allocating_ms"]=timeit(lambda: src.to("cpu").contiguous())
out["copy_into_prealloc_ms"]=timeit(lambda: dst.copy_(src))
print(json.dumps(out, indent=1))

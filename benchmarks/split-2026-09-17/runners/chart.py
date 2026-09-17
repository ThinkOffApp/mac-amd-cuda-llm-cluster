import json, os
from statistics import mean, stdev
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

S = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(S + '/interleaved3.jsonl') if l.strip()]
drows = [json.loads(l) for l in open(S + '/dense.jsonl') if l.strip()]
dPs = [640, 768, 896, 1024, 1280]
dpasses = sorted({r['pass'] for r in drows})
def dratio(P):
    out = []
    for p_ in dpasses:
        a = [r['avg'] for r in drows if r['p'] == P and r['cfg'] == 'split' and r['pass'] == p_]
        b = [r['avg'] for r in drows if r['p'] == P and r['cfg'] == 'gb10' and r['pass'] == p_]
        if a and b and b[0] == b[0] and b[0] > 0: out.append(a[0] / b[0])
    return mean(out), (stdev(out) if len(out) > 1 else 0.0), len(out)
Ps = [128, 512, 1024, 2048, 4096]
passes = sorted({r['pass'] for r in rows})

def val(P, cfg, p):
    v = [r['avg'] for r in rows if r['p'] == P and r['cfg'] == cfg and r['pass'] == p]
    return v[0] if v and v[0] == v[0] else None

def ratio(P, a, b):
    out = []
    for p in passes:
        x, y = val(P, a, p), val(P, b, p)
        if x and y: out.append(x / y)
    return mean(out), (stdev(out) if len(out) > 1 else 0.0), len(out)

def absol(P, cfg):
    v = [val(P, cfg, p) for p in passes]
    v = [x for x in v if x]
    return mean(v), (stdev(v) if len(v) > 1 else 0.0)

fig = plt.figure(figsize=(14, 8.0), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1], wspace=0.2,
                      left=0.062, right=0.978, top=0.755, bottom=0.155)
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for a in (ax, bx):
    a.set_facecolor("#fbfbfd")
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): a.spines[sp].set_color("#c9ccd4")
    a.grid(True, color="#e8eaf0", lw=0.9); a.set_axisbelow(True)
    a.set_xscale("log", base=2); a.set_xticks(Ps)
    a.get_xaxis().set_major_formatter(FuncFormatter(lambda v, p: f"{int(v)}"))
    a.tick_params(colors="#5b6170", labelsize=10.5)
    a.set_xlabel("prompt length (tokens)", fontsize=12, color="#3a3f4b")

# LEFT: ratios
ax.axhspan(1.0, 2.6, color="#eaf5ee", zorder=0)
ax.axhline(1.0, color="#8b93a3", lw=1.3, ls="--", zorder=2)
ax.text(133, 1.035, "above this line the two machines beat the single box",
        fontsize=9.6, color="#5b6170", va="bottom")
for a_, b_, col, lab, mk in (("split", "mac", "#0b6e99", "split  ÷  MacBook alone", "o"),
                             ("split", "gb10", "#d1495b", "split  ÷  GB10 alone", "^")):
    m = [ratio(P, a_, b_) for P in Ps]
    ax.errorbar(Ps, [x[0] for x in m], yerr=[x[1] for x in m], color=col, marker=mk,
                ms=7, lw=2.7, capsize=4, label=lab, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
dm = [dratio(P) for P in dPs]
ax.errorbar(dPs, [x[0] for x in dm], yerr=[x[1] for x in dm], color="#e8871a", marker="D",
            ms=6, lw=2.2, capsize=4, ls=":", label="split ÷ GB10, dense sweep", zorder=5,
            markeredgecolor="white", markeredgewidth=1.1)
ax.set_ylabel("speed relative to one machine", fontsize=12, color="#3a3f4b")
ax.set_ylim(0.72, 2.45)
ax.legend(frameon=False, fontsize=11.5, loc="upper left")
ax.set_title("The split overtakes the faster box between 640 and 896 tokens",
             fontsize=13.5, color="#171a21", pad=11, loc="left")
ax.annotate("near parity at 768 (1.024, s.d. 0.024, n=4);\ntransition bracketed by 640 and 896", xy=(768, 1.024), xytext=(1090, 0.785),
            fontsize=10.0, color="#5b6170", arrowprops=dict(arrowstyle="->", color="#8b93a3", lw=1.2))

# RIGHT: absolutes
for cfg, col, lab, mk in (("mac", "#0b6e99", "MacBook M5 Max (Metal)", "o"),
                          ("gb10", "#7a5195", "NVIDIA GB10 (CUDA)", "s"),
                          ("split", "#d1495b", "both, 50/50 layer split", "^")):
    m = [absol(P, cfg) for P in Ps]
    bx.errorbar(Ps, [x[0] for x in m], yerr=[x[1] for x in m], color=col, marker=mk,
                ms=7, lw=2.7, capsize=4, label=lab, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
bx.set_ylabel("prefill  (tokens / s)", fontsize=12, color="#3a3f4b")
bx.legend(frameon=False, fontsize=11, loc="upper left")
bx.set_title("The Mac fades with context; the GB10 stays flat",
             fontsize=13.5, color="#171a21", pad=11, loc="left")

fig.suptitle("Qwen3.8-27B UD-Q4_K_XL  ·  MacBook Pro M5 Max + ASUS Ascent GX10 (NVIDIA GB10)  ·  llama.cpp RPC layer split over 10 GbE",
             fontsize=12.6, color="#171a21", x=0.06, ha="left", y=0.962)
fig.text(0.062, 0.917,
         "Configurations measured back to back each pass, order rotated.  Circles/triangles n=6, diamonds n=4.  Bars are ± 1 sample s.d. across passes, not a confidence interval.",
         fontsize=10.3, color="#6b7280", ha="left")
fig.text(0.062, 0.880,
         "PREFILL ONLY. Generation is a loss on every split we measured (≈0.68× the Mac alone) — a layer split pipelines, and one token has nothing to overlap.",
         fontsize=10.3, color="#a33", ha="left")
fig.text(0.062, 0.062,
         "Measured under concurrent downloads, so absolute rates are depressed; the in-pass ratios are the robust quantity.",
         fontsize=9.4, color="#9aa0ac", ha="left")
fig.text(0.062, 0.028,
         "Mac build 1d0c76f3c (Metal)  ·  GB10 build 434ddbbc0 (CUDA 13)  ·  direct 10 GbE, 0.6-1.0 ms RTT  ·  17 Sep 2026  ·  github.com/ThinkOffApp/mac-amd-cuda-llm-cluster",
         fontsize=9.4, color="#9aa0ac", ha="left")

out = os.path.join(S, "split-scaling-2026-09-17.png")
fig.savefig(out, facecolor="white")
print("WROTE", out)
for P in Ps:
    r1 = ratio(P, "split", "mac"); r2 = ratio(P, "split", "gb10")
    print(f"  P={P:<5} split/mac {r1[0]:.3f}±{r1[1]:.3f}  split/gb10 {r2[0]:.3f}±{r2[1]:.3f}  n={r1[2]}")

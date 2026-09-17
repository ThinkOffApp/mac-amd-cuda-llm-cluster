import json, os
from statistics import mean, stdev
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "datasets")          # runners/ -> ../datasets
OUT  = os.path.abspath(os.path.join(HERE, ".."))     # provenance directory
def load(f): return [json.loads(l) for l in open(os.path.join(DATA, f)) if l.strip()]

# ---- dense 27B: split vs GB10 (the better box on this model) ----
d3, dn = load('interleaved3.jsonl'), load('dense.jsonl')
def ratio(rows, P, a, b, key='avg'):
    out=[]
    for p in sorted({r['pass'] for r in rows}):
        va=[r[key] for r in rows if r['p']==P and r['cfg']==a and r['pass']==p]
        vb=[r[key] for r in rows if r['p']==P and r['cfg']==b and r['pass']==p]
        if va and vb and vb[0]==vb[0] and vb[0]>0: out.append(va[0]/vb[0])
    return (mean(out), stdev(out) if len(out)>1 else 0.0) if out else None
dense=[]
for P in [128,512,1024,2048,4096]:
    r=ratio(d3,P,'split','gb10')
    if r: dense.append((P,)+r)
for P in [640,768,896,1280]:
    r=ratio(dn,P,'split','gb10')
    if r: dense.append((P,)+r)
dense.sort()

# ---- Flash-Next: split vs Mac (the better box on this model) ----
fn=load('fnsplit.jsonl')
def fparse(v):
    d={}
    for t in v.split():
        k,val=t.split(':'); p,g=k.split('/')
        d[('pp',int(p)) if int(p)>0 else ('tg',int(g))]=float(val)
    return d
def fratio(key,a,b):
    out=[]
    for p in sorted({r['pass'] for r in fn}):
        va=[fparse(r['vals']).get(key) for r in fn if r['cfg']==a and r['pass']==p]
        vb=[fparse(r['vals']).get(key) for r in fn if r['cfg']==b and r['pass']==p]
        if va and vb and va[0] and vb[0]: out.append(va[0]/vb[0])
    return (mean(out), stdev(out) if len(out)>1 else 0.0) if out else None
flash=[(P,)+fratio(('pp',P),'50/50','mac') for P in [512,2048,4096]]

# ---- generation ----
gen=load('gen.jsonl')
def gratio(c,b='mac'):
    out=[]
    for p in sorted({r['pass'] for r in gen}):
        a=[r['tg'] for r in gen if r['cfg']==c and r['pass']==p]
        m=[r['tg'] for r in gen if r['cfg']==b and r['pass']==p]
        if a and m and m[0]>0: out.append(a[0]/m[0])
    return mean(out), (stdev(out) if len(out)>1 else 0.0)
g_dense_5050 = gratio('50/50'); g_dense_1585 = gratio('15/85')
g_flash = fratio(('tg',32),'50/50','mac')

BG="#0e1117"; FG="#e8eaf0"; MUT="#9aa3b2"; GRID="#232936"
A="#4cc2ff"; B="#ff8a5b"; WARN="#ff5d6c"; OK="#57d9a3"
fig=plt.figure(figsize=(18.6,8.8), dpi=150, facecolor=BG)
gs=fig.add_gridspec(1,3,width_ratios=[1.45,0.82,0.95],wspace=0.26,left=0.052,right=0.982,top=0.700,bottom=0.228)
ax,bx,cx=fig.add_subplot(gs[0]),fig.add_subplot(gs[1]),fig.add_subplot(gs[2])
for a_ in (ax,bx):
    a_.set_facecolor(BG)
    for sp in ("top","right"): a_.spines[sp].set_visible(False)
    for sp in ("left","bottom"): a_.spines[sp].set_color("#3a4354")
    a_.grid(True,color=GRID,lw=0.9); a_.set_axisbelow(True)
    a_.tick_params(colors=MUT,labelsize=11)

ax.axhspan(1.0,2.6,color="#12211c",zorder=0)
ax.axhline(1.0,color="#5b6辦".replace("辦","6"),lw=1.4,ls="--",zorder=2)
ax.errorbar([d[0] for d in dense],[d[1] for d in dense],yerr=[d[2] for d in dense],
            color=A,marker="o",ms=7,lw=2.8,capsize=4,zorder=4,mec=BG,mew=1.4,
            label="Dense Qwen3.8-27B · 16.3 GiB · fits either device\n50/50 split ÷ GB10")
ax.errorbar([f[0] for f in flash],[f[1] for f in flash],yerr=[f[2] for f in flash],
            color=B,marker="^",ms=8,lw=2.8,capsize=4,zorder=4,mec=BG,mew=1.4,
            label="Sparse Flash-Next 177B · 87.2 GiB · fits either device\n50/50 split ÷ Mac")
ax.set_xscale("log",base=2); ax.set_xticks([128,512,1024,2048,4096])
ax.get_xaxis().set_major_formatter(FuncFormatter(lambda v,p:f"{int(v)}"))
ax.set_xlabel("prompt length (tokens)",fontsize=12.5,color=FG)
ax.set_ylabel("split  ÷  the better single machine",fontsize=12.5,color=FG)
ax.set_ylim(0.62,1.62)
ax.text(133,1.02,"above the line: two machines beat the best one",fontsize=10.5,color=OK,va="bottom")
ax.text(133,0.665,"reported device budgets:  Mac 107.5 GiB  ·  GB10 121.6 GiB",fontsize=9.6,color=MUT,va="bottom")
ax.text(133,0.965,"below the line: just use the one machine",fontsize=10.5,color=WARN,va="top")
leg=ax.legend(frameon=False,fontsize=10.2,loc="upper left",labelcolor=FG,
              title="both models fit one device, so splitting is a CHOICE   ·   split order: GB10/Mac",title_fontsize=10.5)
leg.get_title().set_color(MUT)
ax.set_title("PREFILL — measured prefill speedup varies with model and prompt length",
             fontsize=13.5,color=FG,pad=12,loc="left")
ax.annotate("near parity at 768",xy=(768,1.024),xytext=(300,1.30),fontsize=10.5,color=A,
            arrowprops=dict(arrowstyle="->",color=A,lw=1.3))
ax.annotate("near parity at 2048",xy=(2048,1.003),xytext=(1150,0.75),fontsize=10.5,color=B,
            arrowprops=dict(arrowstyle="->",color=B,lw=1.3))

labs=["dense 27B\n15/85 · tg64","dense 27B\n50/50 · tg64","Flash-Next\n50/50 · tg32"]
vals=[g_dense_1585[0],g_dense_5050[0],g_flash[0]]
errs=[g_dense_1585[1],g_dense_5050[1],g_flash[1]]
cols=[A,A,B]
bx.axhline(1.0,color="#5b6676",lw=1.4,ls="--",zorder=2)
bx.bar(range(3),vals,yerr=errs,color=cols,width=0.56,zorder=3,capsize=5,
       error_kw=dict(ecolor=MUT,lw=1.4))
for i,v in enumerate(vals):
    bx.text(i,v+0.035,f"{v:.3f}",ha="center",fontsize=12.5,color=FG,zorder=4)
    bx.text(i,v/2,f"−{(1-v)*100:.0f}%",ha="center",va="center",fontsize=13,color=BG,fontweight="bold",zorder=5)
bx.set_xticks(range(3)); bx.set_xticklabels(labs,fontsize=10.5,color=FG)
bx.set_ylim(0,1.18); bx.set_ylabel("split  ÷  the better single machine",fontsize=12.5,color=FG)
bx.set_title("GENERATION — slower for every split we tested.\nThree configurations, two models, one length each.",fontsize=13.5,color=FG,pad=12,loc="left")
bx.text(1.0,1.06,"parity",ha="center",fontsize=10.5,color=MUT)
bx.text(0.5,-0.155,"measured at p=0 (no prompt); generation was NOT swept over prompt length.\nDense is tg64, Flash-Next is tg32 — the two models are not compared\nto each other in this panel.",
        transform=bx.transAxes,ha="center",va="top",fontsize=9.6,color="#c9a227")

cx.axis("off")
cx.set_title("CAPACITY — this 186 GiB quant exceeds\neither device budget",fontsize=13.5,color=FG,pad=12,loc="left")
cx.text(0.0,0.975,"GLM-5.3-Flash  UD-Q4_K_XL",fontsize=13,color=OK,transform=cx.transAxes,va="top",fontweight="bold")
cx.text(0.0,0.905,"186.0 GiB  ·  320.8 B params\nBudgets: Mac 107.5 GiB · GB10 121.6 GiB",
        fontsize=11.2,color=FG,transform=cx.transAxes,va="top")
cx.text(0.0,0.800,"Split 50/50 across both, ONE run, r=2,\n"
                  "not interleaved (see GLM-RUN-PROVENANCE.md):\n"
                  "    prefill   312 t/s @512      424 t/s @2048\n"
                  "    generate  13.7 t/s @32",
        fontsize=11.2,color=FG,transform=cx.transAxes,va="top",family="monospace")
cx.text(0.0,0.612,"The cable does not buy speed here.\nIt buys the model running at all.",
        fontsize=11.6,color=OK,transform=cx.transAxes,va="top")
cx.plot([0.0,1.0],[0.535,0.535],color="#3a4354",lw=1.2,transform=cx.transAxes,clip_on=False)
cx.text(0.0,0.492,"THE LINK  (measured)",fontsize=11.5,color=FG,transform=cx.transAxes,va="top",fontweight="bold")
cx.text(0.0,0.432,"direct 10 GbE, Mac en12 ↔ GB10\n"
                  "9.36 Gbit/s measured on a 17.6 GB file transfer\n"
                  "ICMP round trip 0.6–1.0 ms\n"
                  "Inference wire utilisation was NOT measured.",
        fontsize=10.4,color=MUT,transform=cx.transAxes,va="top")
cx.plot([0.0,1.0],[0.268,0.268],color="#3a4354",lw=1.2,transform=cx.transAxes,clip_on=False)
cx.text(0.0,0.222,"NOT TESTED",fontsize=11.5,color="#c9a227",transform=cx.transAxes,va="top",fontweight="bold")
cx.text(0.0,0.158,"Tensor parallel shares each token's computation\n"
                  "and could potentially speed up generation, if the\n"
                  "compute saved exceeds the communication and\n"
                  "synchronisation it adds.\n\n"
                  "We could not test it: llama.cpp refuses tensor\n"
                  "parallel over RPC (-sm row: \"device RPC0 does not\n"
                  "support split buffers\").\n\n"
                  "Whether a different interconnect would help is\n"
                  "unknown here — it needs engine support and\n"
                  "measurement, neither of which we have.",
        fontsize=10.0,color="#c9a227",transform=cx.transAxes,va="top")

fig.suptitle("Mac + Spark: when does splitting a model across two machines make it faster?",
             fontsize=17,color=FG,x=0.052,ha="left",y=0.962,fontweight="bold")
fig.text(0.052,0.908,"MacBook Pro M5 Max (Metal)  +  ASUS Ascent GX10 / NVIDIA GB10 (CUDA)  ·  llama.cpp RPC layer split over a direct 10 GbE cable",
         fontsize=12,color=MUT,ha="left")
fig.text(0.052,0.862,"Each pass measures every configuration back to back and rotates their order; points are means over n=3–6 passes, bars are ±1 sample s.d. (not a confidence interval).",
         fontsize=10.3,color=MUT,ha="left")
fig.text(0.052,0.822,"Prefill and generation are separate steady-state rates: nothing here measures an end-to-end request, so no total-request speedup can be read off this figure.",
         fontsize=10.3,color="#c9a227",ha="left")
fig.text(0.052,0.060,"Dense prefill measured under concurrent downloads; generation on an idle machine — in-pass ratios are the comparable quantity, absolute rates across panels are not.",
         fontsize=9.6,color="#6f7889",ha="left")
fig.text(0.052,0.030,"Qwen3.8-27B UD-Q4_K_XL  ·  Qwen3.8-Flash-Next UD-IQ4_XS (177 B, 3 B active)  ·  Mac build 1d0c76f3c (Metal), GB10 build 434ddbbc0 (CUDA 13), ggml 0.23.0",
         fontsize=9.6,color="#6f7889",ha="left")
fig.text(0.052,0.008,"17 Sep 2026  ·  data and runners: split-test/provenance-2026-09-17  ·  github.com/ThinkOffApp/mac-amd-cuda-llm-cluster",
         fontsize=9.6,color="#6f7889",ha="left")
out=os.path.join(OUT,"split-when-useful-dark-v3-2026-09-17.png")
fig.savefig(out,facecolor=BG)
print("WROTE",out)
print("dense:",[(d[0],round(d[1],3)) for d in dense])
print("flash:",[(f[0],round(f[1],3)) for f in flash])
print("gen:",round(g_dense_1585[0],3),round(g_dense_5050[0],3),round(g_flash[0],3))

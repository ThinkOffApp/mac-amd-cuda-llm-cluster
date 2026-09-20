#!/bin/bash
# The controlled series petrus asked for: one client, one GGUF, one commit,
# only the layer allocation changes. -ts lists RPC devices FIRST.
#
#   1 mac      Mac alone, Metal, no RPC
#   2 spark1   asus1 only              -ts 1/0      (0 on the Mac)
#   3 sparks2  asus1 + asus2           -ts 1/1/0    (0 on the Mac)
#   4 all3     Mac + asus1 + asus2     -ts 1/1/1
#
# Run 3 still has the Mac orchestrating and sampling; it holds no weights.
set -u
KEY=~/.ssh/id_ed25519_agent_qcd
# MODEL=qwen (default) or MODEL=glm.
#   qwen 87.2 GiB  fits the Mac and one Spark solo -> the milestone series,
#                  because rows 1 and 2 need real fully-resident baselines.
#   glm  186.0 GiB fits NEITHER solo -> capacity/placement study only. There is
#                  no single-host control for it and forcing one would measure
#                  swapping, not the machine. (codexmb, 17 Sep)
case "${MODEL:-qwen}" in
  qwen) MODELF=/Volumes/t705/ModelArchive/Qwen38-Flash-Next-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
        CLIENT=~/llama.cpp/build-rpc/bin/llama-bench ;;
  glm)  MODELF=/Volumes/t705/ModelArchive/GLM-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
        CLIENT=~/llama-glm5/build/bin/llama-bench ;;
  *) echo "MODEL must be qwen or glm"; exit 1 ;;
esac
A1=10.10.10.2; A2=192.168.100.11
OUT=~/split-test/3box; mkdir -p $OUT
P=${P:-512}; N=${N:-128}; R=${R:-3}

case "${1:-}" in
  mac)     TAG=1-mac;      ARGS=() ;;
  spark1)  TAG=2-spark1;   ARGS=(--rpc $A1:50052 -ts 1/0) ;;
  sparks2) TAG=3-sparks2;  ARGS=(--rpc $A1:50052,$A2:50052 -ts 1/1/0) ;;
  all3)    TAG=4-all3;     ARGS=(--rpc $A1:50052,$A2:50052 -ts 1/1/1) ;;
  verify) # cheapest possible run, -v on, purely to capture the buffer allocation lines
     TAG=verify; ARGS=(--rpc $A1:50052,$A2:50052 -ts ${TS_OVERRIDE:-1/1/1} -v) ; P=32; N=1; R=1 ;;
  *) echo "usage: $0 {mac|spark1|sparks2|all3|verify}   [P=512 N=128 R=3]"; exit 1 ;;
esac

TS=$(date -u +%Y%m%dT%H%M%SZ)
J=$OUT/${MODEL:-qwen}-$TAG-p$P-$TS.json; E=$OUT/${MODEL:-qwen}-$TAG-p$P-$TS.err
echo "== ${MODEL:-qwen} $TAG  p=$P n=$N r=$R  model $(basename $MODELF) =="
if ! "$CLIENT" -m "$MODELF" ${ARGS[@]+"${ARGS[@]}"} -p $P -n $N -r $R -o json > "$J" 2> "$E"; then
  echo "  RUN FAILED (exit $?). Last stderr:"; tail -5 "$E"; exit 2
fi
python3 - "$J" <<'PY'
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception as e: print("  UNPARSEABLE RESULT:",e); sys.exit(3)
if not d: print("  EMPTY RESULT - nothing measured"); sys.exit(4)
for r in d:
    kind = f"pp{r['n_prompt']}" if r.get('n_prompt') else f"tg{r['n_gen']}"
    print(f"  {kind:>8}  {r['avg_ts']:8.2f} tok/s  +- {r['stddev_ts']:.2f}")
PY
# An empty verification section is a FAILED check, not a quiet pass. llama-bench
# only emits load_tensors buffer lines with -v, so say so rather than print a
# bare header and let it read as confirmation. (codexmb, 17 Sep)
pl=$(grep -oE "(RPC[0-9]+|Metal|CUDA[0-9]+)[^=]*buffer size *= *[0-9.]+ MiB" "$E" 2>/dev/null | head -8)
if [ -n "$pl" ]; then echo "  placement:"; echo "$pl" | sed 's/^/    /'
else echo "  placement: NO ALLOCATION RECEIPT IN THIS LOG (llama-bench needs -v). Placement is asserted by -ts, not verified here."; fi

#!/bin/bash
# Interleaved matched benchmark: mini-solo, m5-solo, mini+m5 TP.
#
# Configs are interleaved per repetition rather than run in three blocks, so a
# machine drifting over the session perturbs all three equally instead of
# favouring whichever ran while it was cool. This fleet has already produced a
# 15% drift on a Mac across one evening, larger than the effect being measured.
#
# REQUIRES the M5 GPU window to be open (~/m5-gpu-window.sh open claudemm),
# which stops Flash-Next and the room agent. Do not run this without that.
set -u
PROMPT=${1:-128}; NEW=${2:-64}; REPS=${3:-5}
MINI_IP=192.168.50.241
M5=petrus@bosgamem5.local
M5PY=/home/petrus/qwen-drive-venv/bin/python
MINIPY=$HOME/venvs/gpt2tp/bin/python
OUT=/tmp/het-tp/bench-out/p${PROMPT}
mkdir -p "$OUT"

echo "== preflight: M5 GPU window must be open =="
if ! ssh -o BatchMode=yes $M5 'test -f /tmp/m5-gpu-window.open'; then
  echo "ABORT: /tmp/m5-gpu-window.open absent. Flash-Next is serving; timings would be contended." >&2
  exit 2
fi
ssh -o BatchMode=yes $M5 'cat /tmp/m5-gpu-window.open; systemctl is-active llm-server || true'

for r in $(seq 1 "$REPS"); do
  echo "== repetition $r/$REPS =="

  echo "-- mini solo (MPS) --"
  TP_DEVICE=mps $MINIPY /tmp/het-tp/bench.py --mode solo --device mps \
    --prompt-tokens "$PROMPT" --new-tokens "$NEW" --warmups 2 --runs 1 \
    --out "$OUT/mini-solo-$r.json" >/dev/null 2>"$OUT/mini-solo-$r.err" \
    || echo "  mini-solo FAILED (see $OUT/mini-solo-$r.err)"

  echo "-- m5 solo (ROCm) --"
  ssh -o BatchMode=yes $M5 "cd /tmp/het-tp && TP_DEVICE=rocm $M5PY bench.py --mode solo \
    --device rocm --prompt-tokens $PROMPT --new-tokens $NEW --warmups 2 --runs 1" \
    > "$OUT/m5-solo-$r.json" 2>"$OUT/m5-solo-$r.err" \
    || echo "  m5-solo FAILED (see $OUT/m5-solo-$r.err)"

  echo "-- mini + m5 tensor parallel --"
  PORT=$((29900 + r))
  RANK=0 WORLD_SIZE=2 MASTER_ADDR=$MINI_IP MASTER_PORT=$PORT TP_DEVICE=mps \
    $MINIPY /tmp/het-tp/bench.py --mode tp --device mps --prompt-tokens "$PROMPT" \
    --new-tokens "$NEW" --warmups 2 --runs 1 --out "$OUT/tp-$r.json" \
    >/dev/null 2>"$OUT/tp-$r.err" &
  MPID=$!
  sleep 3
  ssh -o BatchMode=yes $M5 "cd /tmp/het-tp && RANK=1 WORLD_SIZE=2 MASTER_ADDR=$MINI_IP \
    MASTER_PORT=$PORT TP_DEVICE=rocm $M5PY bench.py --mode tp --device rocm \
    --prompt-tokens $PROMPT --new-tokens $NEW --warmups 2 --runs 1" \
    >/dev/null 2>"$OUT/tp-m5-$r.err" || echo "  tp rank1 FAILED"
  wait $MPID || echo "  tp rank0 FAILED (see $OUT/tp-$r.err)"
done
echo "== done: $OUT =="

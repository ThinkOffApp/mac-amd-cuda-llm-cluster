#!/bin/bash
# Run @codexmb's rolling-pipeline prototype across the physical Mini<->M5 pair.
#
# WHY A SCRIPT AND NOT A ONE-LINER: the last sweep lost a row silently because
# `cmd | awk ... || echo TIMED OUT` takes its exit status from awk, which
# succeeds on empty input. A blank row then looks exactly like a zero. Here the
# status comes from the command itself, and paging counters are sampled around
# each run so a stall can be diagnosed with deltas rather than with a guess
# about cumulative swap (codexmb was right that a total proves nothing).
set -u

MINI_IP=192.168.50.241
M5=petrus@192.168.50.127
MINI_PY=~/venvs/gpt2tp/bin/python
# Absolute, not ~: the tilde would expand on THIS machine and send the Mini's
# /Users/petrus path to a Linux box that has /home/petrus.
M5_PY=/home/petrus/qwen-drive-venv/bin/python
DIR=/tmp/het-tp
OUT=${OUT:-/tmp/het-tp/rolling-results}
mkdir -p "$OUT"

paging () {  # swapins swapouts pageins pageouts, as a single line of integers
  vm_stat | awk '/Swapins/{a=$2} /Swapouts/{b=$2} /Pageins/{c=$2} /Pageouts/{d=$2}
                 END{gsub(/\./,"",a); gsub(/\./,"",b); gsub(/\./,"",c); gsub(/\./,"",d);
                     print a, b, c, d}'
}

run_one () {
  local label=$1 schedule=$2 batch=$3 chunks=$4 split=$5 port=$6
  local args="--schedule $schedule --batch $batch --chunks $chunks --split $split \
--prompt-tokens 64 --new-tokens 32 --warmups 1 --runs 3"

  read -r s0 so0 p0 po0 <<<"$(paging)"
  local t0 rc
  t0=$(date +%s)

  ( cd "$DIR" && RANK=0 MASTER_ADDR=$MINI_IP MASTER_PORT=$port GPT2_DIR=$DIR/gpt2 \
      $MINI_PY rolling.py --device mps $args --out "$OUT/$label.json" \
      > "$OUT/$label.rank0.log" 2>&1 ) &
  local mini_pid=$!
  sleep 4
  # Status comes from ssh itself, never from a pipeline tail.
  # `timeout` runs on the M5 (Linux) — macOS has no timeout(1), and putting it
  # on this side is a mistake I have now made twice. ConnectTimeout plus the
  # remote timeout bound both halves.
  ssh -o ConnectTimeout=10 $M5 "cd $DIR && timeout 600 env RANK=1 MASTER_ADDR=$MINI_IP \
MASTER_PORT=$port GPT2_DIR=$DIR/gpt2 $M5_PY rolling.py --device rocm $args \
--out $OUT/$label.rank1.json" > "$OUT/$label.rank1.log" 2>&1
  rc=$?
  wait $mini_pid 2>/dev/null
  local elapsed=$(( $(date +%s) - t0 ))

  # The M5 also serves Flash-Next, so it is never idle. Record its load with
  # every row rather than pretending to a clean machine: a relative comparison
  # between schedules survives constant contention, but the reader has to be
  # able to see whether it WAS constant.
  local m5load
  m5load=$(ssh -o ConnectTimeout=5 $M5 "cut -d' ' -f1-3 /proc/loadavg" 2>/dev/null || echo "?")
  read -r s1 so1 p1 po1 <<<"$(paging)"
  local d_si=$((s1-s0)) d_so=$((so1-so0)) d_pi=$((p1-p0)) d_po=$((po1-po0))

  if [ $rc -ne 0 ]; then
    printf "%-22s FAILED rc=%s after %ss   swapin+%d swapout+%d\n" \
      "$label" "$rc" "$elapsed" "$d_si" "$d_so"
    tail -3 "$OUT/$label.rank0.log" | sed 's/^/    rank0: /'
    return 1
  fi

  python3 - "$OUT/$label.json" "$label" "$elapsed" "$d_si" "$d_so" "$d_pi" "$d_po" "$m5load" <<'PY'
import json, sys
path, label, elapsed, si, so, pi, po = sys.argv[1:8]  # sys.argv[8] = m5 load
try:
    r = json.load(open(path))
except Exception as e:
    print(f"{label:<22} NO RESULT FILE ({e})"); raise SystemExit(0)
# rolling.py withholds every rate unless all correctness checks passed, so an
# empty runs list is a FAILED GATE and must never be read as a missing number.
valid = r.get('valid')
rates = [x.get('aggregate_decode_tokens_per_s') for x in r.get('runs', [])]
rates = sorted(v for v in rates if v)
if not valid or not rates:
    print(f"{label:<22} valid={valid!s:<5} NO RATE (gate withheld it)  {elapsed}s  "
          f"swapin+{si} swapout+{so}")
    raise SystemExit(0)
med = rates[len(rates)//2]
print(f"{label:<22} valid={valid!s:<5} decode={med:7.2f} tok/s "
      f"(n={len(rates)}, {rates[0]:.1f}-{rates[-1]:.1f})  {elapsed}s  "
      f"swapin+{si} swapout+{so} pagein+{pi} pageout+{po}  m5load={sys.argv[8]}")
PY
}

case "${1:-calibrate}" in
  calibrate)   # find the layer split that balances THIS pair, per codexmb's README
    echo "== layer-split calibration, alternating, chunks=1, batch 16 =="
    for SP in 2 4 6 8 10; do
      run_one "cal-split$SP" alternating 16 1 "$SP" $((30100+SP))
    done
    ;;
  compare)     # schedules at a fixed split, with the controls he specified
    SP=${2:?usage: run_rolling.sh compare <split>}
    echo "== schedules at split $SP =="
    run_one "c1-whole-b32"  alternating 32 1 "$SP" 30201   # C=1 whole-batch control
    run_one "alt-b32-c4"    alternating 32 4 "$SP" 30202
    run_one "barrier-b32-c4" barrier    32 4 "$SP" 30203
    run_one "rolling-b32-c4" rolling    32 4 "$SP" 30204
    ;;
  *) echo "usage: run_rolling.sh [calibrate|compare <split>]"; exit 2;;
esac

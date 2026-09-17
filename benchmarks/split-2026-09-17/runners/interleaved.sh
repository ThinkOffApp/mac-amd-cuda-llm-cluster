#!/bin/bash
# Interleaved solo/split measurement: drift lands INSIDE the data, not between runs.
set -uo pipefail
S=/private/tmp/claude-501/-Users-petrus/950d475b-a94d-468a-b2ca-15e8c77d8726/scratchpad
M=/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
B=/Users/petrus/llama-glm5/build/bin/llama-bench
OUT=$S/interleaved.jsonl
: > "$OUT"
exec >> $S/interleaved.log 2>&1
echo "=== INTERLEAVED START $(date) ==="
# 3 passes; within each pass every config is measured at every prompt size, configs rotated
for pass in 1 2 3; do
  for P in 512 2048; do
    for cfg in solo 50/50 25/75 15/85; do
      t=$(date +%s)
      if [ "$cfg" = "solo" ]; then
        j=$($B -m "$M" -p $P -n 0 -r 2 -o json 2>/dev/null)
      else
        j=$($B -m "$M" --rpc 10.10.10.2:50052 -ts $cfg -p $P -n 0 -r 2 -o json 2>/dev/null)
      fi
      v=$(printf '%s' "$j" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(f\"{float(d[0]['avg_ts']):.2f} {float(d[0].get('stddev_ts',0)):.2f}\")
except Exception: print('NaN NaN')" 2>/dev/null)
      echo "{\"pass\":$pass,\"p\":$P,\"cfg\":\"$cfg\",\"t\":$t,\"ts\":\"$(date +%H:%M:%S)\",\"avg\":${v% *},\"sd\":${v#* }}" >> "$OUT"
      echo "  pass$pass p$P ${cfg:-solo}: $v"
    done
  done
done
echo "=== INTERLEAVED DONE $(date) ==="

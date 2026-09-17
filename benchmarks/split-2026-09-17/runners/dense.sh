#!/bin/bash
# Dense sweep AROUND the interpolated crossover, order-rotated, to MEASURE it
# instead of interpolating between 512 and 1024 (codexmb, 17 Sep 02:14).
set -uo pipefail
S=/private/tmp/claude-501/-Users-petrus/950d475b-a94d-468a-b2ca-15e8c77d8726/scratchpad
M=/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
B=/Users/petrus/llama-glm5/build/bin/llama-bench
K=/Users/petrus/.ssh/id_ed25519_agent_qcd
OUT=$S/dense.jsonl; : > "$OUT"
exec >> $S/dense.log 2>&1
echo "=== DENSE CROSSOVER SWEEP START $(date) ==="
one() {
  local cfg=$1 P=$2 j
  if [ "$cfg" = "gb10" ]; then
    j=$(ssh -i $K -o BatchMode=yes petrus@10.10.10.2 "LD_LIBRARY_PATH=\$HOME/llama.cpp/build-rpc/bin \$HOME/llama.cpp/build-rpc/bin/llama-bench -m ~/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf -p $P -n 0 -r 3 -ngl 99 -o json 2>/dev/null")
  else j=$($B -m "$M" --rpc 10.10.10.2:50052 -ts 50/50 -p $P -n 0 -r 3 -o json 2>/dev/null); fi
  printf '%s' "$j" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(f\"{float(d[0]['avg_ts']):.2f}\")
except Exception: print('NaN')" 2>/dev/null
}
for pass in 1 2 3 4; do
  if [ $((pass % 2)) -eq 1 ]; then ORDER="split gb10"; else ORDER="gb10 split"; fi
  echo "-- pass $pass order: $ORDER --"
  for P in 640 768 896 1024 1280; do
    for cfg in $ORDER; do
      v=$(one "$cfg" "$P")
      echo "{\"pass\":$pass,\"p\":$P,\"cfg\":\"$cfg\",\"ts\":\"$(date +%H:%M:%S)\",\"avg\":$v}" >> "$OUT"
      echo "   p$P $cfg: $v"
    done
  done
done
echo "=== DENSE DONE $(date) ==="

#!/bin/bash
# Interleaved AND order-rotated: solo(Mac), 50/50 split, gb10(Spark) measured back to back
# at each prompt size, with the config order ROTATED every pass so warmup/drift cannot
# couple to a particular configuration (codexmb, 17 Sep 01:51).
set -uo pipefail
S=/private/tmp/claude-501/-Users-petrus/950d475b-a94d-468a-b2ca-15e8c77d8726/scratchpad
M=/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
B=/Users/petrus/llama-glm5/build/bin/llama-bench
K=/Users/petrus/.ssh/id_ed25519_agent_qcd
OUT=$S/interleaved3.jsonl; : > "$OUT"
exec >> $S/interleaved3.log 2>&1
echo "=== INTERLEAVED-3 (order-rotated) START $(date) ==="
echo "conditions: concurrent downloads active (Flash-Next -> T705, GLM Q6 -> Spark)"
one() {
  local cfg=$1 P=$2 j
  if [ "$cfg" = "mac" ]; then j=$($B -m "$M" -p $P -n 0 -r 2 -o json 2>/dev/null)
  elif [ "$cfg" = "gb10" ]; then
    j=$(ssh -i $K -o BatchMode=yes petrus@10.10.10.2 "LD_LIBRARY_PATH=\$HOME/llama.cpp/build-rpc/bin \$HOME/llama.cpp/build-rpc/bin/llama-bench -m ~/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf -p $P -n 0 -r 2 -ngl 99 -o json 2>/dev/null")
  else j=$($B -m "$M" --rpc 10.10.10.2:50052 -ts 50/50 -p $P -n 0 -r 2 -o json 2>/dev/null); fi
  printf '%s' "$j" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(f\"{float(d[0]['avg_ts']):.2f} {float(d[0].get('stddev_ts',0)):.2f}\")
except Exception: print('NaN NaN')" 2>/dev/null
}
for pass in 1 2 3 4 5 6; do
  case $((pass % 3)) in
    1) ORDER="mac split gb10" ;;
    2) ORDER="split gb10 mac" ;;
    0) ORDER="gb10 mac split" ;;
  esac
  echo "-- pass $pass order: $ORDER --"
  for P in 128 512 1024 2048 4096; do
    for cfg in $ORDER; do
      v=$(one "$cfg" "$P")
      echo "{\"pass\":$pass,\"order\":\"$ORDER\",\"p\":$P,\"cfg\":\"$cfg\",\"ts\":\"$(date +%H:%M:%S)\",\"avg\":${v% *},\"sd\":${v#* }}" >> "$OUT"
      echo "   p$P $cfg: $v"
    done
  done
done
echo "=== INTERLEAVED-3 DONE $(date) ==="

#!/bin/bash
# Generation (tg) at the same rigour as the prefill chart: all configs back to back
# each pass, order rotated, so drift lands inside the data.
set -uo pipefail
S=/private/tmp/claude-501/-Users-petrus/950d475b-a94d-468a-b2ca-15e8c77d8726/scratchpad
M=/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
R=/home/petrus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
B=/Users/petrus/llama-glm5/build/bin/llama-bench
K=/Users/petrus/.ssh/id_ed25519_agent_qcd
OUT=$S/gen.jsonl; : > "$OUT"
exec >> $S/gen.log 2>&1
echo "=== GENERATION INTERLEAVED START $(date) ==="
one() {
  local cfg=$1 j
  case "$cfg" in
    mac)  j=$($B -m "$M" -p 0 -n 64 -r 3 -o json 2>/dev/null) ;;
    gb10) j=$(ssh -i $K -o BatchMode=yes petrus@10.10.10.2 "LD_LIBRARY_PATH=\$HOME/llama.cpp/build-rpc/bin \$HOME/llama.cpp/build-rpc/bin/llama-bench -m $R -p 0 -n 64 -r 3 -ngl 99 -o json 2>/dev/null") ;;
    *)    j=$($B -m "$M" --rpc 10.10.10.2:50052 -ts "$cfg" -p 0 -n 64 -r 3 -o json 2>/dev/null) ;;
  esac
  printf '%s' "$j" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); print(f\"{float(d[0]['avg_ts']):.2f}\")
except Exception: print('NaN')" 2>/dev/null
}
for pass in 1 2 3 4; do
  case $((pass % 4)) in
    1) ORDER="mac 50/50 15/85 gb10" ;;
    2) ORDER="gb10 mac 50/50 15/85" ;;
    3) ORDER="15/85 gb10 mac 50/50" ;;
    0) ORDER="50/50 15/85 gb10 mac" ;;
  esac
  echo "-- pass $pass order: $ORDER --"
  for cfg in $ORDER; do
    v=$(one "$cfg")
    echo "{\"pass\":$pass,\"cfg\":\"$cfg\",\"ts\":\"$(date +%H:%M:%S)\",\"tg\":$v}" >> "$OUT"
    echo "   $cfg: $v"
  done
done
echo "=== DONE $(date) ==="

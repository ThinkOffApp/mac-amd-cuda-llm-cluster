#!/bin/bash
# Flash-Next SPLIT vs both solos, interleaved + order-rotated. Queued behind the
# generation run so they do not contend for the Mac or the link.
set -uo pipefail
S=/private/tmp/claude-501/-Users-petrus/950d475b-a94d-468a-b2ca-15e8c77d8726/scratchpad
M=/Volumes/t705/ModelArchive/Qwen38-Flash-Next-IQ4_XS/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
R=/home/petrus/models/qwen38-flashnext-iq4xs/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
B=/Users/petrus/llama-glm5/build/bin/llama-bench
K=/Users/petrus/.ssh/id_ed25519_agent_qcd
OUT=$S/fnsplit.jsonl; : > "$OUT"
exec >> $S/fnsplit.log 2>&1
while pgrep -f "[g]en_interleaved.sh" >/dev/null; do sleep 20; done
echo "=== FLASH-NEXT SPLIT START $(date) ==="
one() {
  local cfg=$1 j
  case "$cfg" in
    mac)  j=$($B -m "$M" -p 512,2048,4096 -n 32 -r 2 -o json 2>/dev/null) ;;
    gb10) j=$(ssh -i $K -o BatchMode=yes petrus@10.10.10.2 "LD_LIBRARY_PATH=\$HOME/llama.cpp/build-rpc/bin \$HOME/llama.cpp/build-rpc/bin/llama-bench -m $R -p 512,2048,4096 -n 32 -r 2 -ngl 99 -o json 2>/dev/null") ;;
    *)    j=$($B -m "$M" --rpc 10.10.10.2:50052 -ts "$cfg" -p 512,2048,4096 -n 32 -r 2 -o json 2>/dev/null) ;;
  esac
  printf '%s' "$j" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    print(' '.join(f\"{int(x.get('n_prompt',0))}/{int(x.get('n_gen',0))}:{float(x['avg_ts']):.1f}\" for x in d))
except Exception: print('NaN')" 2>/dev/null
}
for pass in 1 2 3; do
  case $((pass % 3)) in
    1) ORDER="mac 50/50 gb10" ;;
    2) ORDER="gb10 mac 50/50" ;;
    0) ORDER="50/50 gb10 mac" ;;
  esac
  echo "-- pass $pass order: $ORDER --"
  for cfg in $ORDER; do
    v=$(one "$cfg")
    echo "{\"pass\":$pass,\"cfg\":\"$cfg\",\"ts\":\"$(date +%H:%M:%S)\",\"vals\":\"$v\"}" >> "$OUT"
    echo "   $cfg: $v"
  done
done
echo "=== DONE $(date) ==="

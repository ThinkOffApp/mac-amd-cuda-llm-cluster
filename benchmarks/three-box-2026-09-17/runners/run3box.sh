#!/bin/bash
# Mac + 2 Spark GLM-5.3-Flash Q4_K_XL over llama.cpp RPC.
#
# Client is the ONLY machine that reads the GGUF: ggml-rpc-server executes
# tensor ops and never parses the model, which is why the two Sparks can run
# stock 434ddbbc0 while the client needs the glm5next build (1d0c76f3c).
# Confirmed by the 16 Sep provenance: "the two ends are DIFFERENT builds".
set -u
KEY=~/.ssh/id_ed25519_agent_qcd
CLIENT=~/llama-glm5/build/bin/llama-bench          # 1d0c76f3c, glm5next-capable
MODEL=/Volumes/t705/ModelArchive/GLM-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf
A1=10.10.10.2          # asus1, direct 10G cable
A2=192.168.100.11      # asus2, reached via asus1 (needs the two routes)
OUT=~/split-test/3box; TS=$(date -u +%Y%m%dT%H%M%SZ)

die(){ echo "FAIL: $*" >&2; exit 1; }

case "${1:-}" in
preflight)
  echo "=== client build actually knows glm5next (probe libllama, NOT llama-bench) ==="
  n=$(strings ~/llama-glm5/build/bin/libllama.dylib 2>/dev/null | grep -c '^glm5next$')
  echo "  libllama.dylib glm5next=$n"; [ "$n" -ge 1 ] || die "client build lacks glm5next"
  c=$(strings ~/llama.cpp/build-rpc/bin/libllama.dylib 2>/dev/null | grep -c '^glm5next$')
  echo "  control, mainline libllama glm5next=$c  (expect 0 - proves the probe discriminates)"
  [ "$c" -eq 0 ] || die "control failed: mainline should NOT have glm5next"
  echo "=== model set complete ==="
  k=$(ls /Volumes/t705/ModelArchive/GLM-Q4_K_XL/*.gguf | wc -l | tr -d ' ')
  echo "  shards=$k (expect 6)"; [ "$k" -eq 6 ] || die "expected 6 shards, found $k"
  [ -r "$MODEL" ] || die "first shard not readable: $MODEL"
  echo "=== both RPC hosts reachable, and by which path ==="
  for h in $A1 $A2; do
    # Test ping's EXIT STATUS. Capturing output and testing for non-empty is
    # not a check: the statistics line is printed even at 100%% packet loss,
    # so the guard could never fire.
    if ping -c 3 -t 3 $h >/tmp/ping.$$ 2>&1; then ok=yes; else ok=no; fi
    echo "  $h  $(tail -1 /tmp/ping.$$)"
    echo "    route: $(route -n get $h 2>/dev/null | awk '/interface|gateway/{printf "%s ", $0}')"
    rm -f /tmp/ping.$$
    [ "$ok" = yes ] || die "$h unreachable - fix the route before running anything"
  done ;;
rpc-up)
  ssh -i $KEY petrus@$A1 'pkill -x ggml-rpc-server; (setsid nohup ~/llama.cpp/build-rpc/bin/ggml-rpc-server -H 10.10.10.2 -p 50052 -d CUDA0 > ~/rpc-3box.log 2>&1 < /dev/null &); sleep 3; pgrep -c -x ggml-rpc-server'
  ssh -i $KEY petrus@$A1 "ssh -o StrictHostKeyChecking=no petrus@$A2 'pkill -x ggml-rpc-server; (setsid nohup ~/llama.cpp/build-rpc/bin/ggml-rpc-server -H $A2 -p 50052 -d CUDA0 > ~/rpc-3box.log 2>&1 < /dev/null &); sleep 3; pgrep -c -x ggml-rpc-server'"
  echo "--- both must answer on 50052 before we call this up ---"
  for h in $A1 $A2; do nc -z -G 3 $h 50052 && echo "  $h:50052 OPEN" || die "$h:50052 NOT listening"; done ;;
run)
  echo "client $(git -C ~/llama-glm5 rev-parse --short HEAD), model $(basename $MODEL)"
  $CLIENT -m "$MODEL" --rpc $A1:50052,$A2:50052 -p 512 -n 128 -r 3 -o json \
    2> $OUT/3box-$TS.err | tee $OUT/3box-$TS.json | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    print(' ', r['n_prompt'] or '', r['n_gen'] or '', 'tok/s', round(r['avg_ts'],2), '+-', round(r['stddev_ts'],2))"
  echo "--- device placement actually used ---"
  grep -E "RPC|rpc|load_tensors|offloaded|buffer size" $OUT/3box-$TS.err | head -15 ;;
rpc-down)
  ssh -i $KEY petrus@$A1 'pkill -x ggml-rpc-server; echo asus1 stopped'
  ssh -i $KEY petrus@$A1 "ssh -o StrictHostKeyChecking=no petrus@$A2 'pkill -x ggml-rpc-server; echo asus2 stopped'" ;;
*) echo "usage: $0 {preflight|rpc-up|run|rpc-down}" ;;
esac

#!/bin/bash
exec /home/petrus/llama.cpp/build-rpc/bin/llama-server \
  -m /home/petrus/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl 999 -c 4096 -np 1 -b 512 -ub 512 -t 4 \
  --host 10.10.10.2 --port 19083 --slot-save-path /tmp/spark-mac-handoff/slots/ \
  --no-webui --log-verbosity 4 --cache-ram 0

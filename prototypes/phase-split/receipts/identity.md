# Raw receipts, cross-machine phase split, 2026-09-18

## Model identity
```
producer (M5, ROCm)  /home/petrus/llm/models/Qwen3.8-27B-UD-Q4_K_XL.gguf
decoder  (Mini, Metal) /Users/petrus/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf
```

## Decoder command line, as captured from the process table
```
./llama-server -m /Users/petrus/models/gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf --slot-save-path /tmp/kvslots2 --port 8091 --host 127.0.0.1 -c 4096 -ngl 99 -np 1
```

## Slot file hashes
```
M5 /tmp/kvprod/pd.bin
4990f010d3a9806f66be91ce45a9ad1012bbdf1cd21442a61e65e164b66ef8e2  /tmp/kvprod/pd.bin
Mini /tmp/kvslots2/
4990f010d3a9806f66be91ce45a9ad1012bbdf1cd21442a61e65e164b66ef8e2  /tmp/kvslots2/pd.bin
7d2f9ed212dd3a94ad4bb1e4b8e9c4aa77c30bd9f3f732d5f9202c9bc1119554  /tmp/kvslots2/q27.bin
```

## GPU window closure / health receipt
```
2026-09-18T13:59:21Z CLOSED: llm-server active health=200, room agent active
window closed
llm-server active health=200; room agent active
```

## Decoder backend identity and memory breakdown
```
0.21.955.269 I cmn          init: llama threadpool init, n_threads = 4
0.23.211.884 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'
```

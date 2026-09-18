#!/bin/bash
# Staging share versus model size, Mini alone. No M5, no window.
for m in gpt2:/tmp/het-tp/gpt2 gpt2-medium:/tmp/het-tp/models/gpt2-medium gpt2-large:/tmp/het-tp/models/gpt2-large; do
  name=${m%%:*}; dir=${m#*:}
  GPT2_DIR=$dir ~/venvs/gpt2tp/bin/python bench.py --mode solo --device mps --stage cpu \
    --prompt-tokens 128 --new-tokens 32 --warmups 1 --runs 1 --profile \
    > sweep_$name.json 2>sweep_$name.err
  echo "  $name exit=$?"
done

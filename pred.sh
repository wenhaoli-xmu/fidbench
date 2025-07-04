#!/bin/bash

max_gen=128

bases=(
    # "qwen2.5-7b-instruct"
    # "qwen2.5-14b-instruct"
    "qwen2.5-32b-instruct"
)

methods=(
    # quant
    # "awq"
    # "gptq"

    # low precision attn
    # "flash-attention-fp8"
    # "sage-attention"

    # kv cache
    # "snapkv"
    # "h2o"
    # "topk"

    # sparse
    "sparsegpt"
    # "wanda"

    # others
    # "transmla"
    # "medusa"
    # "gear"
)

# compression methods
for base in "${bases[@]}"; do

    # baseline 
    test_script="${base}.json"
    echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
    python pred.py --env_conf "runs/${test_script}" --max_gen $max_gen --sample

    for method in "${methods[@]}"; do
        test_script="${base}-${method}.json"

        echo "-----------------------------------"
        echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
        python pred.py --env_conf "runs/${method}/${test_script}" --max_gen $max_gen --label "pred/runs/${base}" --sample
        echo "Finished processing ${test_script}."
        echo "-----------------------------------"

    done
done
#!/bin/bash

max_gen=128

bases=(
    "llama3.1-8b-instruct"
    "qwen2.5-7b-instruct"
)

methods=(
    "topk"
)


# compression methods
for base in "${bases[@]}"; do

    # baseline 
    test_script="${base}.json"
    echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
    python perplexity.py --env_conf "runs/${test_script}"

    for method in "${methods[@]}"; do
        test_script="${base}-${method}.json"

        echo "-----------------------------------"
        echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
        python perplexity.py --env_conf "runs/${test_script}"
        echo "Finished processing ${test_script}."
        echo "-----------------------------------"

    done
done
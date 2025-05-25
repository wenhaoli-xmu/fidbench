#!/bin/bash

max_gen=128

bases=(
    "llama3.1-8b-instruct"
)

methods=(
    "topk"
)


# compression methods
for base in "${bases[@]}"; do

    # baseline 
    test_script="${base}.json"
    echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
    python pred.py --env_conf "runs/${test_script}" --max_gen $max_gen

    for method in "${methods[@]}"; do
        test_script="${base}-${method}.json"

        echo "-----------------------------------"
        echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
        python pred.py --env_conf "runs/${test_script}" --max_gen $max_gen --label "pred/runs/${base}"
        echo "Finished processing ${test_script}."
        echo "-----------------------------------"

    done
done
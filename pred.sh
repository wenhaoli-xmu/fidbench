#!/bin/bash

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
    python pred.py --env_conf "runs/${test_script}" --max_gen 128

    for method in "${methods[@]}"; do
        test_script="${base}-${method}.json"

        echo -e "\033[34mRunning prediction for ${test_script}...\033[0m"
        python pred.py --env_conf "runs/${test_script}" --max_gen 128
        echo "Finished processing ${test_script}."
        echo "-----------------------------------"
        echo -e "\033[34m Results of ${base} [${method}]\033[0m"
        python test_sim.py pred/runs/${base} pred/runs/${base}-${method}
        echo "-----------------------------------"

    done
done
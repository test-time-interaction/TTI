#!/bin/bash

conda activate tti

export VLLM_USE_V1=1
export VLLM_WORKER_USE_RAY=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
python -c "import nltk; nltk.download('punkt_tab')"
export HF_TOKEN=''
export OPENAI_API_KEY=''

# Configuration
NUM_ITERATIONS=15                # Total number of iterations
CONFIG_NAME="webvoyager_rl"       # Configuration name to use

export CURRENT_MAX_STEPS=10

# Main loop for iterations
for ((iteration=1; iteration<=${NUM_ITERATIONS}; iteration++)); do
    echo "============================================"
    echo "Starting iteration ${iteration}/${NUM_ITERATIONS}"
    echo "============================================"

    if [ ${iteration} -eq 1 ]; then
        CURRENT_MAX_STEPS=10
    elif [ ${iteration} -le 3 ]; then
        CURRENT_MAX_STEPS=20 
    else
        CURRENT_MAX_STEPS=30
    fi

    CURRENT_MAX_STEPS_TEST=30 
    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 27999 run.py --config-name ${CONFIG_NAME}  actor_epochs=0 env_config.max_iter=${CURRENT_MAX_STEPS} eval_during_training=false

    sleep 30

    CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --master_port 28999 run.py --config-name ${CONFIG_NAME}  actor_epochs=1 online=false test_env_config.max_iter=${CURRENT_MAX_STEPS_TEST} eval_during_training=true
 
done

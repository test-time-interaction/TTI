#!/bin/bash

conda activate tti

python -c "import nltk; nltk.download('punkt_tab')"
export HF_TOKEN=''
export OPENAI_API_KEY=''

export VLLM_USE_V1=1
export VLLM_WORKER_USE_RAY=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1

PORT=29989

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Running with port: $PORT"
TOKENIZERS_PARALLELISM=false deepspeed --master_port $PORT run.py --config-name webvoyager_eval

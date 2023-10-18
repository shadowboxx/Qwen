#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

MODEL="/mnt/d/LLM/models/Qwen-14B-Chat-Int4" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="finetune/calf8414"

export CUDA_VISIBLE_DEVICES=0

#set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24'

# Remember to use --fp16 instead of --bf16 due to autogptq
## lazy_preprocess 
python3 finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DATA/train.json \
  --bf16 True \
  --output_dir $DATA/output \
  --num_train_epochs 5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 512 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora \
  --q_lora \

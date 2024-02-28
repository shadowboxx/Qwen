#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR="finetune/calf8414"
DATE=$(date +%Y%m%d)

MODEL="/mnt/d/LLM/models/Qwen/Qwen1.5-14B-Chat-GPTQ-Int4" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.

export CUDA_VISIBLE_DEVICES=0

#set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8
#export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24'

# kill api
kill -15 $(pgrep -f openai_api.py)

# Remember to use --fp16 instead of --bf16 due to autogptq
## lazy_preprocess 
python3 finetune.py \
  --model_name_or_path $MODEL \
  --data_path   $DIR/train.json \
  --output_dir  $DIR/output-$DATE \
  --cache_dir   $DIR/cache \
  --num_train_epochs 50 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
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

echo "python3 openai_api.py --checkpoint-path $DIR/output-$DATE" > ./api.sh

./api.sh




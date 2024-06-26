#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR="train/calf8414"
DATE=$(date +%Y%m%d)
OUTPUT=$DIR/output-lora-$DATE

MODEL=/mnt/d/LLM/models/Qwen/Qwen1.5-7B-Chat # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.

export CUDA_VISIBLE_DEVICES=0

# kill api
kill -15 $(pgrep -f openai_api.py)

## lazy_preprocess 
python3 finetune.py \
  --model_name_or_path $MODEL \
  --data_path $DIR/train.json \
  --bf16 True \
  --output_dir $OUTPUT \
  --num_train_epochs 20 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 10 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.95 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "none" \
  --model_max_length 256 \
  --lazy_preprocess True \
  --gradient_checkpointing \
  --use_lora 

if [ $? -eq 0 ]; then
  echo "python3 openai_api.py --checkpoint-path $OUTPUT" > ./api.sh
  ./api.sh
fi




#!/bin/bash

export MODEL_DIR="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="./exps/any2any_tryon"
export TRAIN_DATA_PATH="path/to/jsonl"
export LOG_PATH="$OUTPUT_DIR/logs"

export PYTHONPATH="$(pwd)${PYTHONPATH+":"}${PYTHONPATH-}"

accelerate launch train_lora_any2any.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --source_column="image" \
    --caption_column="prompt" \
    --train_data_path=$TRAIN_DATA_PATH \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --rank=256 \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --num_validation_images=1 \
    --validation_preset="./asset/images/validation_preset.jsonl" \
    --num_train_epochs=100 \
    --validation_steps=250 \
    --checkpointing_steps=1000 \
    --report_to tensorboard \
    --use_8bit_adam

# --gradient_checkpointing

# bash train_lora_any2any.sh 2>&1 | tee -a log.txt
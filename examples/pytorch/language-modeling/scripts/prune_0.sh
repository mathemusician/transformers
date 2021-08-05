#!/bin/bash

GPU=0

ROOT=/hdd/src/neuralmagic/transformers/examples/pytorch/language-modeling

MODEL_DIR=$ROOT/models

#####################

MODEL_NAME=debug

NUM_TRAIN_EPOCHS=1

CUDA_VISIBLE_DEVICES=$GPU python $ROOT/run_mlm.py \
  --model_name_or_path bert-base-uncased \
  --max_train_samples 256 \
  --max_eval_samples 256 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --do_train \
  --do_eval \
  --fp16 \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --max_seq_length 128 \
  --output_dir $MODEL_DIR/$MODEL_NAME \
  --cache_dir /hdd/datasets/huggingface/datasets \
  --preprocessing_num_workers 24 \
  --seed 42 \
  --num_train_epochs $NUM_TRAIN_EPOCHS \
  --save_strategy epoch \
  --save_total_limit 2

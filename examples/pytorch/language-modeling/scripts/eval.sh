#!/bin/bash

GPU=0

ROOT=/hdd/src/neuralmagic/transformers/examples/pytorch/language-modeling

MODEL_DIR=$ROOT/models

#####################

MODEL_NAME=debug

CUDA_VISIBLE_DEVICES=$GPU python $ROOT/run_mlm.py \
  --model_name_or_path $MODEL_DIR/$MODEL_NAME \
  --max_train_samples 64 \
  --max_eval_samples 64 \
  --dataset_name wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --do_eval \
  --per_device_eval_batch_size 4 \
  --max_seq_length 128 \
  --output_dir $MODEL_DIR/$MODEL_NAME/eval \
  --cache_dir /hdd/datasets/huggingface/datasets \
  --preprocessing_num_workers 32 \
  --seed 42

#!/bin/bash

GPU=0,1,2,3

ROOT=/home/tuan/src/neuralmagic/transformers/examples/pytorch/language-modeling

MODEL_DIR=$ROOT/models/mlm

#####################

#MODEL_NAME_OR_PATH=$MODEL_DIR/$MODEL_NAME

MODEL_NAME_OR_PATH=bert-base-uncased

#Wikipedia + Bookcorpus
CUDA_VISIBLE_DEVICES=$GPU python $ROOT/run_mlm.py \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --dataset_name bookcorpus \
  --dataset_name_2 wikitext \
  --dataset_config_name_2 wikitext-103-raw-v1 \
  --validation_split_percentage 5 \
  --do_eval \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --output_dir $MODEL_DIR/$MODEL_NAME_OR_PATH/eval \
  --preprocessing_num_workers 128 \
  --seed 42


####################
# #Wikipedia + Bookcorpus
# CUDA_VISIBLE_DEVICES=$GPU python $ROOT/run_mlm.py \
#   --model_name_or_path $MODEL_NAME_OR_PATH \
#   --dataset_name bookcorpus \
#   --dataset_name_2 wikipedia \
#   --dataset_config_name_2 20200501.en \
#   --validation_split_percentage 5 \
#   --do_eval \
#   --per_device_eval_batch_size 32 \
#   --max_seq_length 128 \
#   --output_dir $MODEL_DIR/$MODEL_NAME_OR_PATH/eval \
#   --preprocessing_num_workers 128 \
#   --seed 42


####################
# # Wikitext
# CUDA_VISIBLE_DEVICES=$GPU python $ROOT/run_mlm.py \
#   --model_name_or_path $MODEL_NAME_OR_PATH \
#   --dataset_name wikitext \
#   --dataset_config_name wikitext-103-raw-v1 \
#   --validation_split_percentage 1 \
#   --do_eval \
#   --per_device_eval_batch_size 16 \
#   --max_seq_length 512 \
#   --output_dir $MODEL_NAME_OR_PATH/eval \
#   --preprocessing_num_workers 32 \
#   --seed 42

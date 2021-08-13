#!/bin/bash

GPU=0

ROOT=/home/tuan/src/neuralmagic/transformers/examples/pytorch/language-modeling

RECIPE_DIR=$ROOT/recipes
MLM_DIR=$ROOT/models/mlm

#####################

DATASETS=bookcorpus_wikitext
MODEL_DIR=$MLM_DIR/$DATASETS


# #########################
# RECIPE_NAME=book_wikitext-prune70_uniform-2epochs
# MODEL_NAME=bert@$RECIPE_NAME

# NUM_TRAIN_EPOCHS=2

# CUDA_VISIBLE_DEVICES=$GPU python $ROOT/run_mlm.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name bookcorpus \
#   --dataset_name_2 wikitext \
#   --dataset_config_name_2 wikitext-103-raw-v1 \
#   --do_train \
#   --do_eval \
#   --fp16 \
#   --logging_steps 2000 \
#   --save_steps 2000 \
#   --per_device_train_batch_size 32 \
#   --per_device_eval_batch_size 32 \
#   --learning_rate 5e-5 \
#   --max_seq_length 512 \
#   --output_dir $MODEL_DIR/$MODEL_NAME \
#   --preprocessing_num_workers 32 \
#   --seed 2021 \
#   --recipe $RECIPE_DIR/$RECIPE_NAME.md \
#   --num_train_epochs $NUM_TRAIN_EPOCHS \
#   --save_total_limit 2

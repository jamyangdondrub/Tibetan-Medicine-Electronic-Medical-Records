#!/bin/env bash


BASE_DIR=/home/yang/Desktop/Tibetan-wwm
BERT_BASE_DIR=$BASE_DIR/bert-tibetan
DATA_DIR=$BASE_DIR/data/cluener_public
OUTPUT_DIR=$BASE_DIR/model
export CUDA_VISIBLE_DEVICES=0

python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 150 \
    --train_file ${DATA_DIR}/train.txt\
    --eval_file ${DATA_DIR}/dev.txt\
    --test_file ${DATA_DIR}/test.txt\
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --num_train_epochs 1\
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR



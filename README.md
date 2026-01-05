# Tibetan-Medicine-Electronic-Medical-Records
Tibetan Medicine Electronic Medical Record Medical Knowledge Unit Recognition

## BERT-BiLSTM-CRF模型
# 输入数据格式请处理成BIO格式，如下：
སྐབས་	O
རེར་	O
མགོ་	O
ཡུ་	O
འཁོར་	O
བ་	O
དང་	O
མགོ་	O
བོ་	O
གཡས་	O
སུ་	O
ན་	B-LB
ཟུག་	I-LB
ལངས་	I-LB
ནས་	I-LB
གཉིད་	I-LB
དཀའ་	I-LB
བ་	E-LB
།	O
མཇིང་	B-BW
ཚིགས་	E-BW
ནས་	O
རོ་	O
སྟོད་	O
བརྒྱུད་	O
དུ་	O
གཟེར་	O
བ་	O
།	O
ཟས་	B-LB
ཀྱི་	I-LB
འཇུ་	I-LB
སྟོབས་	I-LB
ཆུང་	I-LB
ལ་	I-LB
ཟས་	I-LB
ཀྱི་	I-LB
དང་	I-LB
ག་	I-LB
ཞན་	E-LB
།	O
运行的环境
python == 3.7.4
pytorch == 1.3.1 
pytorch-crf == 0.7.2  
pytorch-transformers == 1.2.0               
使用方法
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

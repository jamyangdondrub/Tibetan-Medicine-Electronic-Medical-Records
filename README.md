# 藏医电子病历医学知识单元识别 项目详细文档

## 目录

1. [项目概述](#项目概述)
2. [项目结构](#项目结构)
3. [环境依赖](#环境依赖)
4. [数据格式](#数据格式)
   - [原始数据预处理脚本](#原始数据预处理脚本nert.py-预处理版)
   - [训练数据格式（BIO/BIOES）](#训练数据格式bioes)
5. [各模块详细分析](#各模块详细分析)
   - [models.py — 模型定义](#modelspy--模型定义)
   - [ner.py — 主训练/预测脚本](#nerpy--主训练预测脚本)
   - [utils.py — 数据处理工具](#utilspy--数据处理工具)
   - [conlleval.py — 评估脚本](#conllevalpy--评估脚本)
   - [clue_process.py — CLUE数据集预处理](#clue_processpy--clue数据集预处理)
6. [配置文件说明](#配置文件说明)
7. [使用方法](#使用方法)
8. [模型性能](#模型性能)

---

## 项目概述

本项目实现了一个基于 **TibetanBERT + WWM + BiLSTM + CRF** 的中文命名实体识别系统。模型架构分为三层：

- **TibetanBERT层**：使用预训练的藏文BERT模型对输入文本进行编码，获取上下文相关的字符级表示。
- **BiLSTM层**（可选）：在BERT输出之上叠加双向LSTM，进一步捕获序列上下文信息。
- **CRF层**：条件随机场解码层，利用标签间的转移约束进行全局最优标签序列预测，避免出现非法标签转移（如 `I-PER` 出现在 `B-LOC` 之后）。

---

## 项目结构

```
├── ner.py               # 主入口：训练、评估、测试、推理
├── nert.py              # 原始标注数据→BIOES格式转换脚本
├── models.py            # 模型定义：BERT_BiLSTM_CRF
├── utils.py             # 数据读取、特征转换、Dataset构建
├── conlleval.py         # CoNLL格式评估脚本（计算P/R/F1）
├── clue_process.py      # CLUENER2020数据集JSON→BIO格式转换
├── config.json          # BERT模型配置
├── bert_config.json     # BERT模型配置（备份）
├── vocab.txt            # BERT词表文件
├── train.txt            # 训练数据（BIO格式）
├── dev.txt              # 验证数据（BIO格式）
```

---

## 环境依赖

```
python >= 3.7
pytorch >= 1.3.1
pytorch-crf >= 0.7.2
transformers
pytorch-transformers    
tensorboardX
tqdm
numpy
```

安装命令：

```bash
pip install torch transformers pytorch-crf tensorboardX tqdm numpy pytorch-transformers
```

---

## 数据格式

### 原始标注数据格式

原始数据采用藏文内联标注格式，实体通过 `/实体文本-标签` 的方式直接标记在句子中，非实体部分不加标注。以下是原始数据样例：

```
ན་ལུགས་གཙོ་བོ།གསོ་བྱའི་/མགོ་བོ-BW་/ན-LB་ཞིང་/ལྟག་རྩ-BW་/གཟེར་བ-LB།
/ཡན་ལག་གི་ཚིགས་གཞི-BW་/ན-LB་ཞིང་/བརྐྱང་བསྐུམ་བྱེད་དཀའ་བ-LB།
/མཁལ་རྐེད-BW་/འཁོར་ནས-CD་/ན་བ-LB་བཅས་/ལོ་བཞི-SJ་ལྷག་འགོར།
```

**格式说明**：

- `/མགོ་བོ-BW` 表示"མགོ་བོ"（头部）是一个 **BW**（身体部位）类型的实体
- `/ན-LB` 表示"ན"（疼痛）是一个 **LB**（临床表现）类型的实体
- `/ལོ་བཞི-SJ` 表示"ལོ་བཞི"（四年）是一个 **SJ**（时间）类型的实体
- 没有 `/...-XX` 标记的部分为非实体文本（标签为 O）

**实体类型标签**：

| 标签 | 含义 | 示例 |
|------|------|------|
| BW | 身体部位（Body Part） | མགོ་བོ（头部）、མཁལ་རྐེད（腰肾）、ལྕེ（舌） |
| LB | 症状/病征（Symptom） | ན（痛）、གཟེར་བ（刺痛）、སྐམ་པ（干燥） |
| JB | 疾病名（Disease） | ཡ་སྲིན（痛风）、མཁལ་གཅོང（肾病）、མཆིན་ལྡེམ（肝炎） |
| SJ | 时间（Time） | ལོ་བཞི（四年）、ལོ་སུམ་ཅུ（三十年） |
| CD | 方位/方向（Direction） | འཁོར་ནས（周围） |
| FW | 方位词（Location Word） | མདུན་རྒྱབ（前后）、གཡས་གཡོན（左右） |
| TZ | 检查项目（Test Item） | ལུས་དྲོད（体温）、ཁྲག་ཤེད（血压） |
| JZ | 检查结果（Test Result） | 36.4℃、100:70mmHg |
| JF | 检查方法（Test Method） | ཁྲག་པྲ（血检）、Ct |



**转换效果示例**：

原始标注：`/མགོ་བོ-BW་/ན-LB་ཞིང་`

转换后（BIOES格式）：

```
མགོ	B-BW
བོ	E-BW
ན	S-LB
ཞིང	O
```

### 训练数据格式（BIO/BIOES）

训练数据采用 **BIO/BIOES 标注格式**，每行一个字符及其标签，以制表符分隔，句子之间以空行分隔：

```
彭	B-name
小	I-name
军	I-name
认	O
为	O
，	O
台	B-address
湾	I-address

温	B-name
格	I-name
的	O
```

在本项目的医药领域数据中，标签体系使用了 `b-`/`m-`/`e-` 前缀（即BIOES格式的变体），实体类型包括：`PERSON_GROUP`、`DRUG_TASTE`、`DRUG_DOSAGE`、`DRUG_EFFICACY`、`SYMPTOM` 等。

---

## 使用方法

### 1. 原始藏文数据预处理

将内联标注格式的原始数据转换为BIOES序列标注格式：

```bash
python nert.py
```

输入：`merge_test.txt`（内联标注格式）→ 输出：`ner.txt`（BIOES格式）


### 2. 训练模型

```bash
BERT_BASE_DIR= /path/to/your/model_name
DATA_DIR=./data/
OUTPUT_DIR=./model/output

python ner.py \
    --model_name_or_path ${BERT_BASE_DIR} \
    --do_train True \
    --do_eval True \
    --max_seq_length 256 \
    --train_file ${DATA_DIR}/train.txt \
    --eval_file ${DATA_DIR}/dev.txt \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 10 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir $OUTPUT_DIR
```

### 4. 测试模型

```bash
python ner.py \
    --model_name_or_path /path/to/your/model_name \
    --do_test True \
    --max_seq_length 256 \
    --test_file ./data/test.txt \
    --do_lower_case \
    --need_birnn True \
    --rnn_dim 256 \
    --output_dir ./model/output
```

### 5. 推理（批量预测）

```bash
python ner.py \
    --model_name_or_path /path/to/your/model_name \
    --do_inference True \
    --max_seq_length 512 \
    --test_file ./input_dir/ \
    --do_lower_case \
    --need_birnn True \
    --rnn_dim 256 \
    --output_dir ./model/output
```

### 6. 查看训练日志

```bash
tensorboard --logdir=./model/output/eval
```

---

## 模型性能

在 ZY_MER_Corpus 验证集上的结果：


---


<img width="1849" height="902" alt="image" src="https://github.com/user-attachments/assets/e29c8de5-230d-4de0-bff6-c0aeaefcad7a" />



---


<img width="2559" height="2325" alt="image" src="https://github.com/user-attachments/assets/03bda604-5d14-47b6-8274-1c0dca8f37cd" />

---




## 注意事项

1. **GPU要求**：模型需要CUDA GPU。代码中 `device = torch.device("cuda")` 硬编码了GPU使用，如需CPU运行需手动修改。
2. **预训练模型**：需要下载藏文BERT预训练模型（如 `TibetanBERT`），包含 `config.json`、`vocab.txt`、`pytorch_model.bin`。
3. **长文本处理**：`utils.py` 中的 `read_pred_data` 方法会自动将超过512字符的文本在标点处分割。
4. **标签一致性**：训练和预测时需保证标签体系一致，标签映射保存在 `output_dir/label2id.pkl` 中。
5. **混合导入**：代码同时使用了 `pytorch-transformers`（旧版）和 `transformers`（新版）的导入，实际运行时以 `transformers` 为准。

# 🚀 藏医电子病历医学知识单元识别

一个用于识别**藏医电子病历（EMR）**中医学知识单元的命名实体识别（NER）框架，基于 **TibetanBERT-wwm + BiLSTM + CRF**。

---

## 目录

1. [项目简介](#项目简介)
2. [数据集信息](#数据集信息)
   - [概述与来源](#概述与来源)
   - [数据集统计](#数据集统计)
   - [实体（知识单元）类型](#实体知识单元类型)
   - [数据格式](#数据格式)
   - [数据划分](#数据划分)
   - [数据获取方式](#数据获取方式)
3. [代码信息](#代码信息)
   - [项目结构](#项目结构)
   - [模块说明](#模块说明)
   - [模型配置](#模型配置)
4. [环境依赖](#环境依赖)
5. [使用方法](#使用方法)
   - [安装](#安装)
   - [步骤 1 — 原始数据预处理](#步骤-1--原始数据预处理)
   - [步骤 2 — 训练模型](#步骤-2--训练模型)
   - [步骤 3 — 测试模型](#步骤-3--测试模型)
   - [步骤 4 — 推理（批量预测）](#步骤-4--推理批量预测)
   - [步骤 5 — 查看训练日志](#步骤-5--查看训练日志)
6. [方法流程](#方法流程)
7. [模型性能](#模型性能)
8. [注意事项](#注意事项)
9. [引用](#引用)
10. [许可协议](#许可协议)
11. [贡献指南](#贡献指南)
12. [联系方式](#联系方式)

---

## 项目简介

本仓库提供藏医电子病历**医学知识单元识别**的代码与数据处理流程。针对自由文本形式的藏文临床病历，模型可自动标注十二类医学知识单元（疾病名称、临床表现、身体部位、药物、剂量、生命特征、检查指标等）对应的文本片段。

模型架构由三层堆叠而成：

- **TibetanBERT-wwm 层**——采用全词遮蔽（WWM）预训练的藏文 BERT 模型，对输入文本进行编码，获取上下文相关的音节级表示。
- **BiLSTM 层**（可选）——在 BERT 输出之上叠加双向 LSTM，进一步捕获序列上下文信息。
- **CRF 层**——条件随机场解码层，利用标签间的转移约束进行全局最优标签序列预测，避免非法标签转移（如不兼容的 `B-YY` 之后出现 `I-XX`）。

框架通过单一命令行入口支持训练、评估、测试与批量推理。

---

## 数据集信息

### 概述与来源

模型在 **ZY_MER_Corpus**（藏医电子病历医学实体识别语料库）上训练与评估，这是一个经人工标注的藏医临床病历集合。

- **采集时间段：** 2022–2023
- **语言/文字：** 藏文（乌金体）
- **领域：** 传统藏医学（Sowa Rigpa）电子病历

### 数据集统计

| 项目 | 数量 |
|------|------|
| 电子病历文档数 | 1,487 |
| 标注知识单元总数 | 130,647 |

**各类型实体数量：**

| 标签 | 类型 | 数量 |
|------|------|------|
| JB | 疾病名称 | 8,599 |
| TZ | 生命特征 | 5,751 |
| LB | 临床表现 | 40,609 |
| JZ | 检查指标 | 6,281 |
| YM | 药物名称 | 4,893 |
| JL | 用药剂量 | 4,355 |
| YF | 用药方法 | 5,225 |
| FL | 非药物治疗 | 2,413 |
| BW | 身体部位 | 38,387 |
| FW | 身体方位 | 9,235 |
| CD | 疼痛程度 | 1,550 |
| SJ | 持续时间 | 3,349 |
| **合计** | | **130,647** |

### 实体（知识单元）类型

语料库定义了 **12 类医学知识单元**：

| 标签 | 含义 | 示例 |
|------|------|------|
| TZ | 生命特征 | ལུས་དྲོད（体温）、ཁྲག་ཤེད（血压） |
| JB | 疾病名称 | ཡ་སྲིན（痛风）、མཁལ་གཅོང（肾病）、མཆིན་ལྡེམ（肝炎） |
| LB | 临床表现 | ན（痛）、གཟེར་བ（刺痛）、སྐམ་པ（干燥） |
| JZ | 检查指标 | 36.4℃、100:70mmHg |
| YM | 药物名称 | རིན་ཆེན་མང་སྦྱོར（仁青芒觉）、བྱུ་དམར་ཉེར་ལྔ（二十五味珊瑚） |
| JL | 用药剂量 | རིལ་བུ1（一丸）、ཁེ1（一克） |
| YF | 用药方法 | ཕབ་ཐང（煎汤） |
| FL | 非药物治疗 | བཅིང་ལུམས（热敷）、སྐམ་ཁབ（针灸） |
| BW | 身体部位 | མགོ་བོ（头部）、མཁལ་རྐེད（腰肾）、ལྕེ（舌） |
| FW | 身体方位 | མདུན་རྒྱབ（前后）、གཡས་གཡོན（左右） |
| CD | 疼痛程度 | དྲག་པོ་ན་བ།（剧烈疼痛） |
| SJ | 持续时间 | ལོ་བཞི（四年）、ལོ་སུམ་ཅུ（三十年） |

### 数据格式

**原始标注格式。** 知识单元通过 `/知识单元文本-标签` 的方式直接内联标记在句子中，未标记部分为非实体（标签为 `O`）。示例：

```
ན་ལུགས་གཙོ་བོ།གསོ་བྱའི་/མགོ་བོ-BW་/ན-LB་ཞིང་/ལྟག་རྩ-BW་/གཟེར་བ-LB།/ཡན་ལག་གི་ཚིགས་གཞི-BW་/ན-LB་ཞིང་/བརྐྱང་བསྐུམ་བྱེད་དཀའ་བ-LB།/མཁལ་རྐེད-BW་/འཁོར་ནས-CD་/ན་བ-LB་བཅས་/ལོ་བཞི-SJ་ལྷག་འགོར།
```

- `/མགོ་བོ-BW` → "མགོ་བོ"（头部）是一个 **BW**（身体部位）类型的知识单元
- `/ན-LB` → "ན"（疼痛）是一个 **LB**（临床表现）类型的知识单元
- `/ལོ་བཞི-SJ` → "ལོ་བཞི"（四年）是一个 **SJ**（持续时间）类型的知识单元

**训练格式（BIOES）。** 预处理后，每行存储一个音节及其标签，以制表符分隔（`音节<TAB>标签`），句子之间以空行分隔：

```
མགོ	B-BW
བོ	E-BW
ན	S-LB
ཞིང	O
ལྟག	B-BW
རྩ	E-BW
གཟེར	B-LB
བ	E-LB
།	O
```

转换示例——原始标注 `/མགོ་བོ-BW་/ན-LB་ཞིང་` 转换后为：

```
མགོ	B-BW
བོ	E-BW
ན	S-LB
ཞིང	O
```

所有实体类型均采用 `BIOES` 标注体系：`JB`、`TZ`、`LB`、`JZ`、`YM`、`JL`、`YF`、`FL`、`BW`、`FW`、`CD`、`SJ`。

### 数据划分

数据集按以下比例划分为训练集、验证集与测试集：

| 划分  | 文件  | 比例 |
|------|------|------|
| 训练集 | `data/train.txt`  | 60% |
| 验证集 | `data/dev.txt`    | 20% |
| 测试集 | `data/test.txt`   | 20% |

### 数据获取方式

**ZY_MER_Corpus** 的部分数据已公开：

- 📥 **下载地址：** [https://github.com/jamyangdondrub/ZY-EMR-Corpus](https://github.com/jamyangdondrub/ZY-EMR-Corpus)

请将下载/划分好的文件以 `train.txt`、`dev.txt`、`test.txt` 命名放置于 `data/` 目录下。

---

## 代码信息

### 项目结构

```
Tibetan-WWM/
├── ner.py               # 主入口：训练 / 评估 / 测试 / 推理
├── nert.py              # 将原始内联标注数据转换为 BIOES 格式
├── models.py            # 模型定义：BERT-BiLSTM-CRF
├── utils.py             # 数据处理工具 & Dataset 构建
├── conlleval.py         # 评估脚本（计算 P / R / F1）
├── clue_process.py      # JSON 格式 -> BIO 标注格式转换
│
├── Tibetan-wwm/
│   ├── bert_config.json # BERT 模型配置文件
│   └── vocab.txt        # 词表文件
│
├── data/
│   ├── train.txt        # 训练集（BIOES 格式）
│   ├── dev.txt          # 验证集（BIOES 格式）
│   └── test.txt         # 测试集（BIOES 格式）
│
├── model/
│   ├── logs/            # 训练日志文件
│   └── checkpoints/     # 模型保存权重
│
└── README.md            # 项目说明文件
```

### 模块说明

| 文件 | 说明 |
|------|------|
| **`ner.py`** | 主脚本。通过命令行参数（`--do_train`、`--do_eval`、`--do_test`、`--do_inference`）控制所有流程模式，负责参数解析、数据加载、模型构建、训练循环、模型保存与预测输出。 |
| **`nert.py`** | 预处理。将原始内联标注的藏医电子病历文本转换为 BIOES 序列标注格式。 |
| **`models.py`** | 定义 `BertBiLSTMCRF` 模型类：TibetanBERT 编码器 -> 可选 BiLSTM -> CRF 解码器。 |
| **`utils.py`** | 从 BIOES 文本文件构建 PyTorch `Dataset` 对象；处理分词、标签对齐、`label2id` 映射，并在文本超过 512 字符时于标点处自动切分。 |
| **`conlleval.py`** | 评估。改编自 CoNLL 共享任务，计算每种标签类型及整体的实体级精确率、召回率与 F1 值。 |
| **`clue_process.py`** | 将 CLUE 格式的 JSON 标注转换为 BIO 格式，以兼容本流程。 |

### 模型配置

TibetanBERT 参数（`Tibetan-wwm/config.json`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hidden_size | 768 | 隐藏层维度 |
| num_hidden_layers | 12 | Transformer 编码层数 |
| num_attention_heads | 12 | 多头注意力头数量 |
| intermediate_size | 3072 | FFN 中间层维度 |
| vocab_size | 32000 | 词表大小 |
| max_position_embeddings | 512 | 最大位置编码长度 |
| hidden_dropout_prob | 0.1 | Dropout 概率 |

---

## 环境依赖

```
python           >= 3.7
pytorch          >= 1.3.1
pytorch-crf      >= 0.7.2
transformers
pytorch-transformers
tensorboardX
tqdm
numpy
```

需要**支持 CUDA 的 GPU**（见[注意事项](#注意事项)）。

---

## 使用方法

### 安装

```bash
git clone https://github.com/jamyangdondrub/Tibetan-Medicine-Electronic-Medical-Records.git
cd Tibetan-Medicine-Electronic-Medical-Records
pip install torch transformers pytorch-crf tensorboardX tqdm numpy pytorch-transformers
```

还需下载 **TibetanBERT-wwm 预训练模型**（`config.json`、`vocab.txt`、`pytorch_model.bin`）并放置于本地目录。

### 步骤 1 — 原始数据预处理

将内联标注格式的原始数据转换为 BIOES 序列标注格式：

```bash
python nert.py
```

输入：原始藏医电子病历数据（内联标注格式）-> 输出：BIOES 序列标注文件。

### 步骤 2 — 训练模型

```bash
BERT_BASE_DIR=/path/to/your/model_name
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

### 步骤 3 — 测试模型

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

### 步骤 4 — 推理（批量预测）

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

### 步骤 5 — 查看训练日志

```bash
tensorboard --logdir=./model/output/eval
```

---

## 方法流程

1. **数据预处理**——使用 `nert.py` 将原始内联标注的藏医电子病历文本转换为 BIOES 格式，并划分为训练/验证/测试集。
2. **编码**——TibetanBERT-wwm 生成上下文相关的音节级嵌入表示。
3. **序列建模**——可选的 BiLSTM 在 BERT 输出之上捕获更长距离的序列依赖。
4. **解码**——CRF 层建模标签转移约束，输出全局最优标签序列。
5. **训练**——使用 AdamW 优化器在标注的病历数据上进行端到端微调。
6. **评估**——通过 `conlleval.py` 按 CoNLL 协议计算实体级精确率、召回率与 F1 值。

---

## 模型性能

在 **ZY_MER_Corpus** 验证集上的结果：

| 指标 | 数值 |
|------|------|
| 精确率（Precision） | 92.08% |
| 召回率（Recall） | 94.17% |
| F1 值 | 93.11% |

<img width="1849" height="902" alt="整体结果" src="https://github.com/user-attachments/assets/e29c8de5-230d-4de0-bff6-c0aeaefcad7a" />

<img width="2559" height="2325" alt="各实体类型结果" src="https://github.com/user-attachments/assets/03bda604-5d14-47b6-8274-1c0dca8f37cd" />

---

## 注意事项

1. **GPU 要求。** 代码中 `device = torch.device("cuda")` 硬编码了 GPU 使用，如需 CPU 运行需手动修改此行。
2. **预训练模型。** 训练前需下载藏文 BERT 预训练模型（`config.json`、`vocab.txt`、`pytorch_model.bin`）。
3. **长文本处理。** `utils.py` 中的 `read_pred_data` 方法会自动将超过 512 字符的文本在标点处切分。
4. **标签一致性。** 训练和预测时需保证标签体系一致，标签映射保存在 `output_dir/label2id.pkl` 中。
5. **混合导入。** 代码同时使用了 `pytorch-transformers`（旧版）和 `transformers`（新版）的导入，实际运行时以 `transformers` 为准。

---

## 引用

如果您在研究中使用了本代码或数据集，请引用以下文献：

**TibetanBERT-wwm 预训练模型：**

```bibtex
@article{liang2024tibetanbertwwm,
  author  = {Liang, Yiming and Lv, Hua and Li, Ya and Duo, La and Liu, Cili and Zhou, Qiang},
  title   = {Tibetan-BERT-wwm: A Tibetan Pretrained Model With Whole Word Masking for Text Classification},
  journal = {IEEE Transactions on Computational Social Systems},
  volume  = {11},
  number  = {5},
  pages   = {6268--6277},
  year    = {2024},
  doi     = {10.1109/TCSS.2024.3374633}
}
```

**ZY_MER_Corpus 数据集：**

```bibtex
@misc{zy_mer_corpus,
  author       = {jamyangdondrub},
  title        = {ZY-EMR-Corpus: Tibetan Electronic Medical Record Corpus},
  year         = {2024},
  howpublished = {\url{https://github.com/jamyangdondrub/ZY-EMR-Corpus}}
}
```

---

## 许可协议

本项目基于 [MIT 许可协议](LICENSE) 发布。学术与科研用途可免费使用；商业用途请联系作者。

---

## 贡献指南

欢迎贡献：

1. Fork 本仓库。
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 提交 Pull Request。

---

## 联系方式

如有问题，请联系 **jamyangdondrub**（邮箱：**zwxxzx@qhnu.edu.cn**），或在 GitHub 上提交 Issue。

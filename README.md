[English](README.md) | [简体中文](README-CN.md)


# 🚀 Tibetan Electronic Medical Record — Medical Knowledge Unit Recognition

A named-entity recognition (NER) framework for identifying medical knowledge units in **Tibetan-medicine electronic medical records (EMRs)**, based on **TibetanBERT-wwm + BiLSTM + CRF**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
   - [Overview & Provenance](#overview--provenance)
   - [Dataset Statistics](#dataset-statistics)
   - [Entity (Knowledge-Unit) Types](#entity-knowledge-unit-types)
   - [Data Format](#data-format)
   - [Data Splits](#data-splits)
   - [How to Obtain the Data](#how-to-obtain-the-data)
3. [Code Information](#code-information)
   - [Project Structure](#project-structure)
   - [Module Descriptions](#module-descriptions)
   - [Model Configuration](#model-configuration)
4. [Requirements](#requirements)
5. [Usage Instructions](#usage-instructions)
   - [Installation](#installation)
   - [Step 1 — Preprocess Raw Data](#step-1--preprocess-raw-data)
   - [Step 2 — Train the Model](#step-2--train-the-model)
   - [Step 3 — Test the Model](#step-3--test-the-model)
   - [Step 4 — Inference (Batch Prediction)](#step-4--inference-batch-prediction)
   - [Step 5 — View Training Logs](#step-5--view-training-logs)
6. [Methodology](#methodology)
7. [Model Performance](#model-performance)
8. [Notes](#notes)
9. [Citations](#citations)
10. [License](#license)
11. [Contributing](#contributing)
12. [Contact](#contact)

---

## Project Overview

This repository provides the code and data-processing pipeline for **medical knowledge-unit recognition** in Tibetan-medicine EMRs. Given free-text Tibetan clinical records, the model automatically labels spans corresponding to twelve categories of medical knowledge units (disease names, clinical manifestations, body parts, drugs, dosages, vital signs, examination indicators, etc.).

The model architecture consists of three stacked layers:

- **TibetanBERT-wwm layer** — a pre-trained Tibetan BERT with whole-word masking (WWM) that encodes the input text into context-aware, syllable-level representations.
- **BiLSTM layer** *(optional)* — a bidirectional LSTM stacked on top of the BERT output to further capture sequential context.
- **CRF layer** — a Conditional Random Field decoder that applies label-transition constraints to produce the globally optimal label sequence, preventing illegal transitions (e.g., an incompatible `B-YY` followed by `I-XX`).

The framework supports training, evaluation, testing, and batch inference through a single command-line entry point.

---

## Dataset Information

### Overview & Provenance

The model is trained and evaluated on **ZY_MER_Corpus** (Tibetan Electronic Medical Record — Medical Entity Recognition corpus), a manually annotated collection of Tibetan-medicine clinical records.

- **Collection period:** 2022–2023
- **Language / script:** Tibetan (Uchen script)
- **Domain:** Traditional Tibetan medicine (Sowa Rigpa) electronic medical records

### Dataset Statistics

| Item | Count |
|------|-------|
| EMR documents | 1,487 |
| Total annotated knowledge units | 130,647 |

**Entity counts by type:**

| Label | Type | Count |
|-------|------|-------|
| JB | Disease Name | 8,599 |
| TZ | Vital Signs | 5,751 |
| LB | Clinical Manifestation | 40,609 |
| JZ | Examination Indicators | 6,281 |
| YM | Drug Name | 4,893 |
| JL | Drug Dosage | 4,355 |
| YF | Administration Method | 5,225 |
| FL | Non-Drug Treatment | 2,413 |
| BW | Body Part | 38,387 |
| FW | Body Location/Direction | 9,235 |
| CD | Pain Intensity | 1,550 |
| SJ | Duration | 3,349 |
| **Total** | | **130,647** |

### Entity (Knowledge-Unit) Types

The corpus defines **12 medical knowledge-unit categories**:

| Label | Meaning | Example |
|-------|---------|---------|
| TZ | Vital Signs | ལུས་དྲོད (body temperature), ཁྲག་ཤེད (blood pressure) |
| JB | Disease Name | ཡ་སྲིན (gout), མཁལ་གཅོང (kidney disease), མཆིན་ལྡེམ (hepatitis) |
| LB | Clinical Manifestation | ན (pain), གཟེར་བ (stabbing pain), སྐམ་པ (dryness) |
| JZ | Examination Indicators | 36.4℃, 100:70mmHg |
| YM | Drug Name | རིན་ཆེན་མང་སྦྱོར (Rinchen Mangjor), བྱུ་དམར་ཉེར་ལྔ (Twenty-Five Coral) |
| JL | Drug Dosage | རིལ་བུ1 (1 pill), ཁེ1 (1 gram) |
| YF | Administration Method | ཕབ་ཐང (decoction) |
| FL | Non-Drug Treatment | བཅིང་ལུམས (hot compress), སྐམ་ཁབ (acupuncture) |
| BW | Body Part | མགོ་བོ (head), མཁལ་རྐེད (lumbar/kidney), ལྕེ (tongue) |
| FW | Body Location/Direction | མདུན་རྒྱབ (front/back), གཡས་གཡོན (left/right) |
| CD | Pain Intensity | དྲག་པོ་ན་བ། (severe pain) |
| SJ | Duration | ལོ་བཞི (four years), ལོ་སུམ་ཅུ (thirty years) |

### Data Format

**Raw annotation format.** Knowledge units are marked inline within sentences using the pattern `/knowledge_unit_text-label`. Unmarked text is treated as non-entity (label `O`). Example:

```
ན་ལུགས་གཙོ་བོ།གསོ་བྱའི་/མགོ་བོ-BW་/ན-LB་ཞིང་/ལྟག་རྩ-BW་/གཟེར་བ-LB།/ཡན་ལག་གི་ཚིགས་གཞི-BW་/ན-LB་ཞིང་/བརྐྱང་བསྐུམ་བྱེད་དཀའ་བ-LB།/མཁལ་རྐེད-BW་/འཁོར་ནས-CD་/ན་བ-LB་བཅས་/ལོ་བཞི-SJ་ལྷག་འགོར།
```

- `/མགོ་བོ-BW` → "མགོ་བོ" (head) is a **BW** (body part) knowledge unit
- `/ན-LB` → "ན" (pain) is an **LB** (clinical manifestation) knowledge unit
- `/ལོ་བཞི-SJ` → "ལོ་བཞི" (four years) is an **SJ** (duration) knowledge unit

**Training format (BIOES).** After preprocessing, data is stored one syllable per line, `syllable<TAB>label`, with sentences separated by blank lines:

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

Conversion example — raw `/མགོ་བོ-BW་/ན-LB་ཞིང་` becomes:

```
མགོ	B-BW
བོ	E-BW
ན	S-LB
ཞིང	O
```

All entity types use the `BIOES` scheme: `JB`, `TZ`, `LB`, `JZ`, `YM`, `JL`, `YF`, `FL`, `BW`, `FW`, `CD`, `SJ`.

### Data Splits

The dataset is split into training, validation, and test sets in the following proportions:

| Split | File | Proportion |
|-------|------|------------|
| Training | `data/train.txt` | 60% |
| Validation | `data/dev.txt` | 20% |
| Test | `data/test.txt` | 20% |

### How to Obtain the Data

A subset of **ZY_MER_Corpus** is publicly available:

- 📥 **Download:** [https://github.com/jamyangdondrub/ZY-EMR-Corpus](https://github.com/jamyangdondrub/ZY-EMR-Corpus)

Place the downloaded/split files under the `data/` directory as `train.txt`, `dev.txt`, and `test.txt`.

---

## Code Information

### Project Structure

```
Tibetan-WWM/
├── ner.py               # Main entry point: training / evaluation / testing / inference
├── nert.py              # Converts raw inline-annotated data to BIOES format
├── models.py            # Model definition: BERT-BiLSTM-CRF
├── utils.py             # Data-processing utilities & Dataset construction
├── conlleval.py         # Evaluation script (computes Precision / Recall / F1)
├── clue_process.py      # JSON -> BIO annotation-format conversion
│
├── Tibetan-wwm/
│   ├── bert_config.json # BERT model configuration file
│   └── vocab.txt        # Vocabulary file
│
├── data/
│   ├── train.txt        # Training set (BIOES format)
│   ├── dev.txt          # Validation set (BIOES format)
│   └── test.txt         # Test set (BIOES format)
│
├── model/
│   ├── logs/            # Training log files
│   └── checkpoints/     # Saved model weights
│
└── README.md            # Project documentation
```

### Module Descriptions

| File | Description |
|------|-------------|
| **`ner.py`** | Main script. Controls all pipeline modes via command-line flags (`--do_train`, `--do_eval`, `--do_test`, `--do_inference`). Handles argument parsing, data loading, model construction, the training loop, checkpointing, and prediction output. |
| **`nert.py`** | Preprocessing. Converts raw inline-annotated Tibetan EMR text into BIOES sequence-labeling format. |
| **`models.py`** | Defines the `BertBiLSTMCRF` model class: TibetanBERT encoder -> optional BiLSTM -> CRF decoder. |
| **`utils.py`** | Builds PyTorch `Dataset` objects from BIOES text files; handles tokenization, label alignment, `label2id` mapping, and automatic splitting of texts longer than 512 characters at punctuation. |
| **`conlleval.py`** | Evaluation. Adapted from the CoNLL shared task; computes entity-level Precision, Recall, and F1 per label type and overall. |
| **`clue_process.py`** | Converts CLUE-format JSON annotations to BIO format for pipeline compatibility. |

### Model Configuration

TibetanBERT parameters (`Tibetan-wwm/config.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_size | 768 | Hidden-layer dimension |
| num_hidden_layers | 12 | Number of Transformer encoder layers |
| num_attention_heads | 12 | Number of attention heads |
| intermediate_size | 3072 | FFN intermediate-layer dimension |
| vocab_size | 32000 | Vocabulary size |
| max_position_embeddings | 512 | Maximum position-encoding length |
| hidden_dropout_prob | 0.1 | Dropout probability |

---

## Requirements

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

A **CUDA-capable GPU** is required (see [Notes](#notes)).

---

## Usage Instructions

### Installation

```bash
git clone https://github.com/jamyangdondrub/Tibetan-Medicine-Electronic-Medical-Records.git
cd Tibetan-Medicine-Electronic-Medical-Records
pip install torch transformers pytorch-crf tensorboardX tqdm numpy pytorch-transformers
```

You also need the **TibetanBERT-wwm pre-trained model** (`config.json`, `vocab.txt`, `pytorch_model.bin`) placed in a local directory.

### Step 1 — Preprocess Raw Data

Convert raw inline-annotated data to BIOES sequence-labeling format:

```bash
python nert.py
```

Input: raw Tibetan EMR data (inline annotation) -> Output: BIOES sequence-labeling file.

### Step 2 — Train the Model

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

### Step 3 — Test the Model

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

### Step 4 — Inference (Batch Prediction)

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

### Step 5 — View Training Logs

```bash
tensorboard --logdir=./model/output/eval
```

---

## Methodology

1. **Data preprocessing** — Raw inline-annotated Tibetan EMR text is converted to BIOES format with `nert.py`, then split into training/validation/test sets.
2. **Encoding** — TibetanBERT-wwm produces context-aware syllable-level embeddings.
3. **Sequence modeling** — An optional BiLSTM captures longer-range sequential dependencies over the BERT outputs.
4. **Decoding** — A CRF layer models label-transition constraints and outputs the globally optimal label sequence.
5. **Training** — The model is fine-tuned end-to-end with the AdamW optimizer on the labeled EMR data.
6. **Evaluation** — Entity-level Precision, Recall, and F1 are computed with the CoNLL protocol via `conlleval.py`.

---

## Model Performance

Results on the **ZY_MER_Corpus** validation set:

| Metric | Value |
|--------|-------|
| Precision | 92.08% |
| Recall | 94.17% |
| F1 | 93.11% |

<img width="1849" height="902" alt="overall-results" src="https://github.com/user-attachments/assets/e29c8de5-230d-4de0-bff6-c0aeaefcad7a" />

<img width="2559" height="2325" alt="per-entity-results" src="https://github.com/user-attachments/assets/03bda604-5d14-47b6-8274-1c0dca8f37cd" />

---

## Notes

1. **GPU requirement.** The code hardcodes GPU usage via `device = torch.device("cuda")`. To run on CPU, modify this line manually.
2. **Pre-trained model.** Download the Tibetan BERT model (`config.json`, `vocab.txt`, `pytorch_model.bin`) before training.
3. **Long-text handling.** The `read_pred_data` method in `utils.py` automatically splits texts longer than 512 characters at punctuation marks.
4. **Label consistency.** The label scheme must be identical between training and prediction; the mapping is saved in `output_dir/label2id.pkl`.
5. **Mixed imports.** The code imports from both `pytorch-transformers` (legacy) and `transformers` (current); at runtime, `transformers` takes precedence.

---

## Citations

If you use this code or dataset in your research, please cite:

**TibetanBERT-wwm pre-trained model:**

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

**ZY_MER_Corpus dataset:**

```bibtex
@misc{zy_mer_corpus,
  author       = {jamyangdondrub},
  title        = {ZY-EMR-Corpus: Tibetan Electronic Medical Record Corpus},
  year         = {2024},
  howpublished = {\url{https://github.com/jamyangdondrub/ZY-EMR-Corpus}}
}
```

---

## License

This project is released under the [MIT License](LICENSE). It is free to use for academic and research purposes. For commercial use, please contact the authors.

---

## Contributing

Contributions are welcome:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request.

---

## Contact

For questions, please contact **jamyangdondrub** (email: **zwxxzx@qhnu.edu.cn**), or open an issue on GitHub.

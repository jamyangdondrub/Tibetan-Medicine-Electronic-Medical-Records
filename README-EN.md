# 🚀 Tibetan Electronic Medical Record Medical Knowledge Unit Recognition: Detailed Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Environment Dependencies](#environment-dependencies)
4. [Methodology](#methodology)
5. [Data Format](#data-format)
   - [Raw Data Preprocessing Script](#raw-data-preprocessing-script-nertpy-preprocessing-version)
   - [Training Data Format (BIOES)](#training-data-format-bioes)
6. [Detailed Module Analysis](#detailed-module-analysis)
   - [models.py — Model Definition](#modelspy--model-definition)
   - [ner.py — Main Training/Prediction Script](#nerpy--main-trainingprediction-script)
   - [utils.py — Data Processing Utilities](#utilspy--data-processing-utilities)
   - [conlleval.py — Evaluation Script](#conllevalpy--evaluation-script)
   - [clue_process.py — CLUE Dataset Preprocessing](#clue_processpy--clue-dataset-preprocessing)
7. [Configuration File Description](#configuration-file-description)
8. [Usage](#usage)
9. [Model Performance](#model-performance)
10. [Citations](#citations)
11. [License](#license)
12. [Contributing](#contributing)

---

## Project Overview

This project implements a medical knowledge unit recognition model for Tibetan electronic medical records based on **TibetanBERT + WWM + BiLSTM + CRF**. The model architecture consists of three layers:

- **TibetanBERT Layer**: Uses a pre-trained Tibetan BERT model to encode input text and obtain context-aware character-level representations.
- **BiLSTM Layer** (optional): Stacks a bidirectional LSTM on top of the BERT output to further capture sequential context information.
- **CRF Layer**: A Conditional Random Field decoding layer that applies transition constraints between labels to predict globally optimal label sequences, preventing illegal label transitions (e.g., `I-PER` appearing after `B-LOC`).

---

## 📌 Project Structure

```
Tibetan-WWM/
├── ner.py               # Main entry point: training / evaluation / testing / inference
├── nert.py              # Converts raw annotated data to BIOES format
├── models.py            # Model definition: BERT-BiLSTM-CRF
├── utils.py             # Data processing utilities & Dataset construction
├── conlleval.py         # Evaluation script (computes P / R / F1)
├── clue_process.py      # JSON format → BIO annotation format conversion
├── Tibetan-wwm/
│   ├── bert_config.json  # BERT model configuration file
│   ├── vocab.txt         # Vocabulary file
├── data/
│   ├── train.txt         # Training set (BIOES annotation format)
│   ├── dev.txt           # Validation set (BIOES annotation format)
│   ├── test.txt          # Test set (BIOES annotation format)
├── model/
│   ├── logs/             # Training log files
│   ├── checkpoints/      # Saved model weights
└── README.md             # Project documentation
```

---

## Environment Dependencies

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

Installation command:

```bash
pip install torch transformers pytorch-crf tensorboardX tqdm numpy pytorch-transformers
```

---

## Methodology

1. **Data Preprocessing** — Raw inline-annotated Tibetan EMR text is converted to BIOES format using `nert.py`.
2. **Model Architecture** — TibetanBERT encodes token embeddings → optional BiLSTM captures sequential context → CRF decodes globally optimal labels.
3. **Training** — Fine-tuned with AdamW optimizer on labeled EMR data.
4. **Evaluation** — Entity-level F1 score using CoNLL protocol via `conlleval.py`.

---

## Data Format

### Raw Annotation Data Format

The raw data uses an inline annotation format, where knowledge units are directly marked within sentences using the pattern `/knowledge_unit_text-label`. Non-knowledge-unit portions are left unannotated. Below is a sample of the raw data:

```
ན་ལུགས་གཙོ་བོ།གསོ་བྱའི་/མགོ་བོ-BW་/ན-LB་ཞིང་/ལྟག་རྩ-BW་/གཟེར་བ-LB།/ཡན་ལག་གི་ཚིགས་གཞི-BW་/ན-LB་ཞིང་/བརྐྱང་བསྐུམ་བྱེད་དཀའ་བ-LB།/མཁལ་རྐེད-BW་/འཁོར་ནས-CD་/ན་བ-LB་བཅས་/ལོ་བཞི-SJ་ལྷག་འགོར།
ད་ལྟའི་ནད་ཀྱི་ལོ་རྒྱུས།ནད་2002གཞིའི་སློང་རྐྱེན་04མི་གསལ་07བར་ན་ལུགས་སུ་/མགོ་བོ-BW་དང་/མིག་ཕྲུམ-BW་/ན་བ-LB།/སྣ་གདོང-BW་/གཟེར-LBལ་/སྣ-BW་/འཚང་བ-LB།/སྣ་ནང-BW་/སྐམ་པ-LB།སྐབས་རེར་/མགོ་ཡུ་འཁོར-LB་བ་དང་/མགོ་བོ-BW་གཡས་སུ་/ན་ཟུག་ལངས-LB་ནས་/གཉིད་དཀའ་བ-LB།
/མཇིང་ཚིགས-BW་ནས་/རོ་སྟོད-BW་བརྒྱུད་དུ་/གཟེར་བ-LB།/ཟས་ཀྱི་འཇུ་སྟོབས་ཆུང-LB་ལ་/ཟས་ཀྱི་དང་ག་ཞན-LB།ཞོགས་པར་/ཁ་ཏིག་ཁ་བ-LB།/ཁ-BW་/ལྕེ-BW་/སྐམ-LB་ལ་/སྐོམ་དད་ཆེ་བ-LB།/མཆིན་པ-BW་/མཁྲིས་བ-BW་/མདུན་རྒྱབ-FW་ཏུ་/སྦོས་ནས་ན་བ-LB།
```

**Format description**:

- `/མགོ་བོ-BW` means "མགོ་བོ" (head) is a knowledge unit of type **BW** (body part)
- `/ན-LB` means "ན" (pain) is a knowledge unit of type **LB** (clinical manifestation)
- `/ལོ་བཞི-SJ` means "ལོ་བཞི" (four years) is a knowledge unit of type **SJ** (duration)
- Portions without a `/...-XX` marker are non-knowledge-unit text (labeled as O)

**Knowledge Unit Type Labels**:

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

**Conversion Example**:

Raw annotation: `/མགོ་བོ-BW་/ན-LB་ཞིང་`

After conversion (BIOES format):

```
མགོ	B-BW
བོ	E-BW
ན	S-LB
ཞིང	O
```

### Training Data Format (BIOES)

The training data uses the **BIOES** annotation format. Each line contains one character and its corresponding label, separated by a tab character. Sentences are separated by blank lines:

```
མགོ B-BW
བོ E-BW
ན S-LB
ཞིང O
ལྟག B-BW
རྩ E-BW
གཟེར B-LB
བ E-LB
། O
ཡན B-BW
ལག I-BW
གི I-BW
ཚིགས I-BW
གཞི E-BW
ན S-LB
ཞིང O
བརྐྱང B-LB
བསྐུམ I-LB
བྱེད I-LB
དཀའ I-LB
བ E-LB
། O
```

In the medical domain data of this project, the label system uses the `BIOES` format, and the entity types include: `JB`, `TZ`, `LB`, `JZ`, `YM`, `JL`, `YF`, `FL`, `BW`, `FW`, `CD`, `SJ`, etc.

---

## Detailed Module Analysis

### models.py — Model Definition

Defines the `BertBiLSTMCRF` model class. Stacks TibetanBERT → optional BiLSTM → CRF decoder.

### ner.py — Main Training/Prediction Script

Main script controlling all pipeline modes via command-line flags: `--do_train`, `--do_eval`, `--do_test`, `--do_inference`.

### utils.py — Data Processing Utilities

Builds PyTorch `Dataset` objects from BIOES text files. Handles tokenization, label alignment, and automatic sequence splitting at punctuation for texts exceeding 512 characters.

### conlleval.py — Evaluation Script

Adapted from the CoNLL shared task. Computes entity-level Precision, Recall, and F1 per label type and overall.

### clue_process.py — CLUE Dataset Preprocessing

Converts CLUE-format JSON annotations to BIO format for pipeline compatibility.

---

## Configuration File Description

### 📌 TibetanBERT Configuration Parameters (`Tibetan-wwm/config.json`)

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| hidden_size | 768 | Hidden layer dimension |
| num_hidden_layers | 12 | Number of Transformer encoder layers |
| num_attention_heads | 12 | Number of multi-head attention heads |
| intermediate_size | 3072 | FFN intermediate layer dimension |
| vocab_size | 32000 | Vocabulary size |
| max_position_embeddings | 512 | Maximum position encoding length |
| hidden_dropout_prob | 0.1 | Dropout probability |

---

## Usage

### 1. Raw Tibetan Data Preprocessing

Convert raw data from inline annotation format to BIOES sequence labeling format:

```bash
python nert.py
```

Input: `Raw Tibetan electronic medical record data` (inline annotation format) → Output: `Sequence labeling file` (BIOES format)

### 2. Train the Model

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

### 3. Test the Model

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

### 4. Inference (Batch Prediction)

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

### 5. View Training Logs

```bash
tensorboard --logdir=./model/output/eval
```

---

## Model Performance

### Results on the ZY_MER_Corpus Validation Set:

---

<img width="1849" height="902" alt="image" src="https://github.com/user-attachments/assets/e29c8de5-230d-4de0-bff6-c0aeaefcad7a" />

---

<img width="2559" height="2325" alt="image" src="https://github.com/user-attachments/assets/03bda604-5d14-47b6-8274-1c0dca8f37cd" />

---

# 🔗 Raw Data: ZY_MER_Corpus

- ## 📥 Raw data: [Download](https://github.com/jamyangdondrub/ZY-EMR-Corpus)

> **Note:** A subset of ZY_MER_Corpus is publicly available at the link above. 
---

## Notes

1. **GPU Requirement**: The model requires a CUDA-capable GPU. The code hardcodes GPU usage with `device = torch.device("cuda")`. If you need to run on CPU, you must manually modify this.
2. **Pre-trained Model**: You need to download the Tibetan BERT pre-trained model (e.g., `TibetanBERT`), which includes `config.json`, `vocab.txt`, and `pytorch_model.bin`.
3. **Long Text Handling**: The `read_pred_data` method in `utils.py` will automatically split texts exceeding 512 characters at punctuation marks.
4. **Label Consistency**: The label scheme must remain consistent between training and prediction. The label mapping is saved in `output_dir/label2id.pkl`.
5. **Mixed Imports**: The code uses both `pytorch-transformers` (legacy) and `transformers` (current) imports. In practice, `transformers` takes precedence at runtime.

---

## Citations

If you use this code or dataset in your research, please cite the following:

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

This project is released under the [MIT License](LICENSE).
Free to use for academic and research purposes.
For commercial use, please contact the authors.

---

## Contributing

Contributions are welcome!

1. Fork this repository: `https://github.com/jamyangdondrub/Tibetan-Medicine-Electronic-Medical-Records/fork`
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request at: `https://github.com/jamyangdondrub/Tibetan-Medicine-Electronic-Medical-Records/pulls`

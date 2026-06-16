# Emotion Classification in Indonesian E-Commerce Reviews

A comparative study of three models for emotion classification on Indonesian-language
product reviews: **XLM-R**, **IndoBERT**, and **BiLSTM+FastText**.

---

## Project Structure
├── data/

│   ├── raw/

│   │   └── PRDECT-ID_Dataset.csv         # Raw dataset

│   └── processed/

│       ├── train.csv

│       ├── val.csv

│       ├── test.csv

│       ├── label_mapping.json

│       ├── class_weights.json

│       └── bilstm_metadata.json

├── notebooks/

│   ├── 00_EDA.ipynb                      # Exploratory data analysis

│   ├── 01_preprocessing.ipynb            # Preprocessing & data splitting

│   ├── 02_xlm-r.ipynb                    # XLM-RoBERTa model

│   ├── 03_indobert.ipynb                 # IndoBERT model

│   ├── 04_bilstm-fasttext.ipynb          # BiLSTM + FastText model

│   └── 05_comparison.ipynb               # Cross-model comparison

├── results/

│   ├── xlmr_predictions.csv

│   ├── indobert_predictions.csv

│   ├── bilstm_fasttext_predictions.csv

│   ├── final_comparison.csv

│   └── comparison_dashboard.png

└── README.md

---

## Dataset

This project uses the **PRDECT-ID Dataset** — a publicly available collection of
Indonesian product reviews from Tokopedia, annotated with emotion labels by a group
of annotators following criteria established by a clinical psychology expert.

- **Emotion classes:** Anger, Fear, Happy, Love, Sadness
- **Coverage:** 29 product categories
- **Language:** Indonesian (Bahasa Indonesia)
- **License:** CC BY 4.0

> Sutoyo, R., Chowanda, A., Achmad, S., Andangsari, E. W., Isa, S. M.,
> Soetandar, J. P., Andres, J. R., Richard, R., Hadiwijaya, M. A., Andrews, D.,
> & Pirdaus, F. P. (2022). *Product Reviews Dataset for Emotions Classification
> Tasks - Indonesian (PRDECT-ID) Dataset* (Version 1). Mendeley Data.
> https://doi.org/10.17632/574v66hf2v.1

---

## Models

### 1. XLM-RoBERTa (XLM-R)
A multilingual transformer trained on 100 languages, used here as a strong
multilingual baseline.

- **Base model:** `xlm-roberta-base`
- **Framework:** HuggingFace Transformers + PyTorch
- **Training:** 5 epochs, lr=1.5e-5, batch size 8
- **Class imbalance:** Weighted loss via custom Trainer

### 2. IndoBERT
A BERT model pretrained specifically on Indonesian corpora. Chosen for its
linguistic alignment with the dataset language.

- **Base model:** `indobenchmark/indobert-base-p1`
- **Framework:** HuggingFace Transformers + PyTorch
- **Training:** 5 epochs, lr=3e-5, batch size 16
- **Class imbalance:** Weighted loss via custom Trainer

### 3. BiLSTM + FastText
A sequential deep learning model using bidirectional LSTM with Indonesian
FastText word embeddings.

- **Embeddings:** Indonesian FastText (`cc.id.300.bin`), 300 dimensions
- **Framework:** PyTorch
- **Training:** Early stopping (stopped at epoch 19, best val F1 at epoch 14)
- **Class imbalance:** Class-weighted loss function

---

## Results

Evaluated on the test set (n=526). **Macro F1 is the primary metric** given the
uneven class distribution.

### Aggregate Metrics

| Model              | Accuracy | Weighted F1 | Macro F1 |
|--------------------|----------|-------------|----------|
| XLM-R              | 0.7072   | 0.7082      | 0.6845   |
| **IndoBERT**       | **0.7205** | **0.7160** | **0.6878** |
| BiLSTM+FastText    | 0.5760   | 0.5774      | 0.5380   |

### Per-Class F1-Score

| Class   | XLM-R  | IndoBERT | BiLSTM+FastText |
|---------|--------|----------|-----------------|
| Anger   | 0.6515 | 0.6032   | 0.5217          |
| Fear    | 0.5371 | 0.5350   | 0.2825          |
| Happy   | 0.8338 | 0.8415   | 0.7836          |
| Love    | 0.7333 | 0.7529   | 0.5750          |
| Sadness | 0.6667 | 0.7063   | 0.5271          |

### Key Findings

- **IndoBERT achieves the best overall performance** across all metrics, which
  is consistent with it being pretrained on Indonesian text.
- **XLM-R is competitive** despite being a general multilingual model, trailing
  IndoBERT by a small margin on all metrics.
- **BiLSTM+FastText performs significantly worse**, particularly on the *Fear*
  class (F1 = 0.28), indicating the model's limitation in handling lexically
  ambiguous short texts without full bidirectional contextual attention.
- **Happy** is the easiest class for all three models; **Fear** and **Anger**
  are the hardest — likely due to lexical overlap between negative emotion classes.

---

## Setup

### Requirements

```bash
pip install transformers torch scikit-learn pandas numpy matplotlib
```

For the BiLSTM notebook, also install FastText:

```bash
pip install fasttext
```

Then download the Indonesian embedding inside the notebook:

```python
import fasttext.util
fasttext.util.download_model('id', if_exists='ignore')
```

### Notebook Execution Order

00_EDA.ipynb              → Data exploration

01_preprocessing.ipynb    → Preprocessing & train/val/test split

02_xlm-r.ipynb            → XLM-R training & evaluation

03_indobert.ipynb         → IndoBERT training & evaluation

04_bilstm-fasttext.ipynb  → BiLSTM+FastText training & evaluation

05_comparison.ipynb       → Cross-model comparison & visualisation

> Notebooks 02–04 are recommended to run on Kaggle (GPU T4) due to
> computational requirements. Notebook 05 can be run locally.

---

## Contributors

**Data & Preprocessing**
* Zahra' Zakiyyah Priyono

**Model Engineering (XLM-R, IndoBERT, BiLSTM+FastText)**
* Zahra' Zakiyyah Priyono
* Angelina Jolie Candaya

**Evaluation & Analysis**
* Zahra' Zakiyyah Priyono

**Research Paper & Documentation**
* Zahra' Zakiyyah Priyono
* Angelina Jolie Candaya
* Maureen Calista Surjo

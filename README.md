# DSS5104-CA2: Text Classification Benchmark

## Overview

This project benchmarks text classification methods **from classical NLP baselines to fine-tuned Transformers**, comparing performance, data efficiency, and cost-accuracy trade-offs across two contrasting datasets:

- **AG News** (4-class topic classification) — where classical methods are expected to be competitive
- **Tweet Eval Irony** (binary irony detection) — where Transformers should outperform due to contextual understanding

## Models

| Tier | Model | Library |
|------|-------|---------|
| Tier 1 (Classical) | TF-IDF + Logistic Regression | scikit-learn |
| Tier 1 (Classical) | TF-IDF + Linear SVM | scikit-learn |
| Tier 2 (Neural) | TextCNN | PyTorch |
| Tier 3 (Transformer) | DistilBERT | HuggingFace Transformers |
| Tier 3 (Transformer) | RoBERTa-base | HuggingFace Transformers |
| Tier 3 (Transformer) | SetFit (few-shot, 1%/5%/10% only) | SetFit |

## Key Experiments

1. **Full benchmark** — All models trained on 100% data with 3 random seeds (42, 123, 456)
2. **Data efficiency study** — Learning curves at 100%, 50%, 25%, 10%, 5%, 1% data fractions on both datasets
3. **Cost-accuracy trade-off** — Training time vs. Macro F1 scatter plots
4. **Error analysis** — 30 misclassified samples categorized by failure mode (ambiguous labels, short text, sarcasm, cross-topic vocab, label noise)

## Quick Start

### Google Colab (Recommended)

1. Upload `text_classification_benchmark.ipynb` to Colab
2. Set runtime to **GPU (T4)** — estimated ~3-5 hours for full run
3. Run all cells sequentially

### Local (GPU required)

```bash
pip install -r requirements.txt
jupyter notebook text_classification_benchmark.ipynb
```

> **Note:** Running on CPU only is not recommended (~50+ hours). A CUDA-capable GPU is strongly advised.

## Project Structure

```
DSS5104-CA2/
├── text_classification_benchmark.ipynb   # Main notebook (all code + analysis)
├── requirements.txt                      # Python dependencies
└── README.md
```

## Evaluation Metrics

- Overall accuracy (mean +/- std across 3 seeds)
- Macro-averaged F1 score
- Per-class F1 for best/worst models
- Training and inference time

## Reproducibility

- All experiments use 3 random seeds: 42, 123, 456
- Results reported as mean +/- std
- Hyperparameters selected on **validation set only**
- Test set used **exactly once** for final reporting
- `requirements.txt` included for environment reproducibility

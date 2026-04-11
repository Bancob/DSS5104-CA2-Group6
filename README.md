# DSS5104 CA2 — Text Classification: Classical NLP to Transformers

**Group 6** — Ban Xiaomeng · Xu Xinyi · Wang Jiadi · Ye Chulin

**Repository:** https://github.com/Bancob/DSS5104-CA2-Group6
**Assignment spec:** https://alexxthiery.github.io/teaching/DSS5104/text-classification.html

This repository contains a complete, reproducible end-to-end text-classification benchmark that compares three tiers of methods (classical TF-IDF baselines, a compact convolutional neural network, and pretrained Transformers, plus SetFit for few-shot) on two deliberately contrasting datasets. The written report is **[`DSS5104 CA2-Group6-Text Classification Classical NLP to Transformers.pdf`](./DSS5104%20CA2-Group6-Text%20Classification%20Classical%20NLP%20to%20Transformers.pdf)**; every number in the report can be reproduced from `dss5104_ca2_checkpoint.pkl` / `experiment_results_full.csv` in this repository.

---

## 1. Datasets

Two datasets chosen to contrast where classical and pretrained methods each shine:

| Dataset | Source | Train / Val / Test | Classes | Text length (words, median / 95th) | Hypothesis |
|---|---|---|---|---|---|
| **AG News** | `ag_news` on HF Datasets | 102,000 / 7,600 / 7,600 | 4 (World / Sports / Business / Sci-Tech) | 37 / 53 | TF-IDF should be competitive (topic vocabulary is distinctive). |
| **Tweet Eval — Irony** | `tweet_eval` → `irony` on HF Datasets | 2,862 / 955 / 784 | 2 (Non-Irony / Irony) | 12 / 23 | Transformers should dominate (irony requires contextual understanding). |

A `train / validation / test` split is used for every experiment. The test split is evaluated **exactly once** per `(model, dataset, seed, data_fraction)` combination, at the very end. Hyperparameters are selected on the validation split only.

---

## 2. Models

| Tier | Model | Library | HP search space |
|---|---|---|---|
| **Tier 1 — Classical** | TF-IDF + Logistic Regression | scikit-learn | `C ∈ {0.1, 1, 10}` × `n-gram ∈ {(1,1), (1,2)}` |
| **Tier 1 — Classical** | TF-IDF + Linear SVM | scikit-learn | same grid as above |
| **Tier 2 — Neural** | TextCNN (3/4/5-gram filters, 100 each) | PyTorch | `LR ∈ {1e-3, 5e-4}`, 10 epochs, early stopping (patience=3) |
| **Tier 3 — Transformer** | DistilBERT-base-uncased | HuggingFace Transformers | `LR ∈ {1e-5, 2e-5, 5e-5}`, 4 epochs, early stopping (patience=2) |
| **Tier 3 — Transformer** | RoBERTa-base | HuggingFace Transformers | same as DistilBERT |
| **Tier 3 — Few-shot** | SetFit (paraphrase-MiniLM-L6-v2) | SetFit | default; **run only at data fractions 1% / 5% / 10%** per assignment |

Transformer `max_seq_len = 128` subword tokens (covers > 99% of AG News and 100% of Tweet Irony without truncation). TextCNN uses a separate word-level `max_seq_len = 256`.

---

## 3. Experiments conducted

All experiments run with **3 random seeds (42, 123, 456)**, as required by the assignment ("at least 3 different random seeds"). Results reported as `mean ± std`.

1. **Part 5 — Full-data benchmark (100% training data).** All 5 core models × 2 datasets × 3 seeds = 30 evaluations. Full LR hyperparameter search on every (model, dataset, seed) triple.
2. **Part 6 — Data-efficiency sweep.** All 5 core models + SetFit × 2 datasets × 3 seeds × 6 data fractions `{100%, 50%, 25%, 10%, 5%, 1%}` (SetFit only at 1/5/10% per assignment). 168 additional evaluations. Stratified sampling preserves class proportions at every fraction. Learning rates cached from Part 5 are reused at every fraction via `fixed_lr=` so the learning curves isolate the effect of *data size* from HP-search noise.
3. **Cost–accuracy trade-off.** Training and inference times recorded for every run; Pareto frontier analysis in the report.
4. **Error analysis.** 30 misclassified test examples per model manually categorised (ambiguous labels, short text, OOD vocabulary, cross-topic overlap, sarcasm miscue), plus full-test-set error overlap statistics between best and worst models on each dataset.

Total: **198 single-seed test evaluations** (cross-validation-verified against the checkpoint).

---

## 4. Repository structure

```
DSS5104-CA2-Group6/
├── DSS5104 CA2-Group6-Text Classification ... .pdf  # Written report (final submission)
├── DSS5104 CA2-Group6-Text Classification ... .docx # Written report (editable source)
├── DSS5104_text_classification_benchmark-vF.ipynb   # Main notebook (with final Colab outputs)
├── dss5104_ca2_checkpoint.pkl                       # Pickled collector + BEST_LR_CACHE (198 results)
├── experiment_results_full.csv                      # Flattened 198-row results table
├── figures/                                         # Figures used in the report (PNG, 300 dpi)
├── requirements.txt                                 # Pinned Python dependencies
└── README.md                                        # This file
```

---

## 5. Reproducing the results

### Option A — Google Colab (recommended)

1. Upload **[DSS5104_text_classification_benchmark-vF.ipynb](DSS5104_text_classification_benchmark-vF.ipynb)** to Colab. (You can also clear all outputs first via `Edit → Clear all outputs` if you prefer to start fresh.)
2. Set runtime to **T4** (or L4 if you have Colab Pro). `Runtime → Change runtime type → T4 GPU`.
3. Run every cell **from top to bottom**.
   - The first install cell (`Cell 3`) pins the HF ecosystem and numpy. After it finishes: **`Runtime → Restart session`**, then continue.
   - `Cell 5` auto-mounts Google Drive and sets `CHECKPOINT_PATH = /content/drive/MyDrive/dss5104_ca2_checkpoint.pkl`. Authorise when prompted.
   - Every training cell is **idempotent**: if a checkpoint already contains a `(model, dataset, seed, data_fraction)` result, that combination is logged as `[skip]` and not retrained. This makes the notebook resumable across the Colab 24-hour session limit.
4. Expected runtime for a clean run: **~25 h on L4** or **~40 h on T4**, distributed across 3–4 Colab sessions. See the "Resuming Tier 3" markdown cell inside the notebook for a concrete per-session plan.

### Option B — Local machine (GPU strongly recommended)

```bash
git clone https://github.com/Bancob/DSS5104-CA2-Group6.git
cd DSS5104-CA2-Group6
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook DSS5104_text_classification_benchmark-vF.ipynb
```

CPU-only execution is not recommended (~50+ h). A CUDA-capable GPU is strongly advised. Tier 1 models always run on CPU via scikit-learn; Tier 2/3 models auto-detect CUDA via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

### Option C — Inspect results without retraining

Load the shipped checkpoint directly to reproduce every number in the report without running any training:

```python
import pickle
with open("dss5104_ca2_checkpoint.pkl", "rb") as f:
    data = pickle.load(f)
# data["results"]        -> list[ExperimentResult] with 198 entries
# data["best_lr_cache"]  -> dict[(model, dataset) -> best LR from Part 5]
```

Or work from the flattened CSV:

```python
import pandas as pd
df = pd.read_csv("experiment_results_full.csv")  # 198 rows
```

---

## 6. Key compute-cost optimisations (assignment-compliant)

Two optimisations are applied that **preserve the full assignment methodology** (all 6 fractions and ≥ 3 seeds remain):

- **Best-LR caching across fractions.** The LR hyperparameter search over `{1e-5, 2e-5, 5e-5}` is performed once on 100% data in Part 5 (as required by the assignment). The winning LR per `(model, dataset)` is cached in `BEST_LR_CACHE` and reused at every Part 6 fraction via the `fixed_lr=` argument. This is standard learning-curve practice — it isolates the effect of training-set size from HP-search noise — and saves ~60% of Part 6 transformer compute. The Part 5 benchmark itself still performs the full three-LR search.
- **Reduced transformer `max_seq_len` from 256 → 128.** AG News 95th-percentile length is ~70 subword tokens, so 128 covers > 99% of samples with no meaningful truncation; this cuts Tier 3 compute by ~40%. TextCNN's word-level `max_seq_len` is unchanged at 256.

---

## 7. Reproducibility and test-set integrity

- **Random seeds:** `42, 123, 456` (three seeds, as required).
- **Validation protocol:** every hyperparameter choice (C / n-gram for TF-IDF, LR for TextCNN, LR for transformers) is made on the validation split only. The test set is evaluated exactly once per `(model, dataset, seed, data_fraction)` combination at the end, via the `.predict(test_texts)` call inside each `train_*()` function.
- **Stratified sampling** preserves class proportions at every Part 6 fraction (`sklearn.model_selection.train_test_split(..., stratify=labels)`).
- **Environment pinning:** see [requirements.txt](requirements.txt). `numpy<2` is required because the HF ecosystem locked at `transformers==4.41.2` pre-dates the NumPy 2.0 C-API changes.
- **Checkpoint integrity:** the 198 entries in `dss5104_ca2_checkpoint.pkl` exactly match the numbers reported in the PDF report; run the script at the end of Section 5 Option C to verify.
- **Idempotency:** re-running any cell cannot corrupt results. The idempotent `has_result()` check, combined with the Part-6 cleanup logic that drops any fractions left in an inconsistent state by a crashed run, guarantees that a re-run either reproduces bit-for-bit (deterministic models) or falls within the reported std (stochastic models).

---

## 8. Notable findings (summary; full discussion in the PDF report)

- **AG News:** TF-IDF + SVM reaches 91.8% accuracy; RoBERTa reaches 95.1%. The 3.3 pp gap comes at ~30× training cost and ~1,700× inference latency. TF-IDF + SVM is the Pareto-rational default for AG News.
- **Tweet Irony:** RoBERTa reaches 74.1% vs 64.5% for TF-IDF + SVM, a 9.5 pp gap that makes RoBERTa worth the ~90 s training cost.
- **Data efficiency:** no crossover on AG News (RoBERTa leads at every fraction from 1% to 100%). On Tweet Irony, **SetFit is the best model at ≤ 10% data**; fine-tuned transformers only dominate above 25% data.
- **Error analysis:** on AG News, 85% of errors for both tiers are intrinsic label ambiguity (cleaning labels would help more than upgrading models). On Tweet Irony, the two tiers fail on substantially complementary examples (only 1/3 overlap), suggesting a simple ensemble could add several pp.

---

## 9. License and acknowledgments

This project was produced for DSS5104 at NUS under the supervision of Prof. Alex Thiery. Dataset terms follow the respective HuggingFace dataset cards (`ag_news`, `tweet_eval`). Model weights follow the licenses of `distilbert-base-uncased`, `roberta-base`, and `paraphrase-MiniLM-L6-v2`.

# LoRA vs Full Fine-Tuning: A Controlled Perplexity Study on GPT-2

This project presents a controlled comparison between **full fine-tuning** and **Low-Rank Adaptation (LoRA)** for causal language models. We study how parameter-efficient fine-tuning behaves under constrained hardware settings, using **perplexity** as the primary evaluation metric.

The goal is not to “beat” full fine-tuning, but to understand **when, why, and how** LoRA approaches its performance — and where it falls short.

---

## Motivation

As language models scale, full fine-tuning becomes increasingly expensive in terms of:
- GPU memory
- Training time
- Optimizer state size

LoRA proposes a solution by freezing the base model and learning low-rank updates to selected weight matrices. This project empirically evaluates that trade-off in a **clean, reproducible setup**.

---

## Experimental Setup

### Model
- **GPT-2 Small (124M parameters)**
- Causal language modeling objective

> GPT-2 Medium full fine-tuning exceeded available GPU memory (6 GB), so GPT-2 Small was chosen to ensure reproducibility and fairness.

---

### Dataset
- **OpenAssistant Conversations (OASST1)**
- Assistant responses only
- Treated as instruction-following SFT data
- Tokenized to a fixed sequence length

Dataset preparation is fully automated via script.

---

### Training Methods

#### 1. Full Fine-Tuning
- All model parameters updated
- AdamW optimizer
- Mixed precision (FP16)

#### 2. LoRA Fine-Tuning
- Base model frozen
- Trainable low-rank adapters applied to:
  - Attention projections (`c_attn`)
  - Attention output projection (`c_proj`)
- Configurable rank `r` and scaling factor `α`
- Significantly fewer trainable parameters (<5%)

---

### Evaluation Metric

- **Perplexity (PPL)** on held-out samples
- Computed as:

\[
\text{PPL} = \exp\left(\frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i \right)
\]

where \(\mathcal{L}_i\) is the causal language modeling loss.

---

## Results

| Method | Trainable Parameters | Perplexity |
|------|----------------------|------------|
| Full Fine-Tuning | ~124M | **13.97** |
| LoRA (r=8) | ~1–2M | ~25.7 |
| LoRA (r=32) | ~4–5M | **~21.9** |

### Key Observations
- Low-rank LoRA (r=8) significantly underfits instruction-following data
- Increasing LoRA rank improves perplexity substantially
- Even at higher ranks, LoRA does not fully match full fine-tuning under identical training budgets
- Learning rate scaling is critical for LoRA optimization

---

## Project Structure
lora-vs-finetuning/
│
├── data/
│ └── prepare_oasst.py # Dataset preprocessing
│
├── training/
│ ├── train_full_ft.py # Full fine-tuning script
│ └── train_lora.py # LoRA fine-tuning script
│
├── evaluation/
│ └── perplexity.py # Perplexity evaluation
│
├── requirements.txt
├── .gitignore
└── README.md



Model checkpoints and processed datasets are intentionally excluded for reproducibility and size constraints.

---

## How to Run

### 1. Setup environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Prepare Dataset
```bash
python data/prepare_oasst.py
```
### 3. Train Models
```bash
python training/train_full_ft.py
python training/train_lora.py
```
### 4. Evaluate Perplexity
```bash
python evaluation/perplexity.py
```


















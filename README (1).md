# AI Customer Support Ticket Triage

An AI system that reads incoming support tickets, predicts their **category** and **urgency**, and routes them to the appropriate queue — flagging low-confidence predictions for human review.

## Overview

Support teams receive a high volume of tickets that need to be manually read, categorized, and prioritized before they reach the right agent. This project automates that first triage step using two independent text classifiers trained on ticket text.

**Pipeline:** Data → Text Cleaning → TF-IDF Features → Two Classifiers (Category + Urgency) → Evaluation → FastAPI Backend → Streamlit Dashboard

## Architecture

```
ticket-triage/
├── models/
│   ├── category_model.pkl    # Predicts: Billing / Technical / Cancellation / Product / Refund
│   ├── urgency_model.pkl     # Predicts: Low / Medium / High / Critical
│   └── vectorizer.pkl        # Shared TF-IDF vectorizer
├── api.py                    # FastAPI backend serving /predict
├── dashboard.py               # Streamlit UI for testing tickets interactively
└── requirements.txt
```

- **Feature extraction:** TF-IDF (unigrams + bigrams, 10k max features)
- **Models:** Logistic Regression (`class_weight='balanced'`) — one for category, one for urgency, sharing the same TF-IDF vectorizer
- **Confidence-based routing:** predictions below a 0.5 confidence threshold on either model are flagged `needs_human_review: true` instead of being auto-routed

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Running

**1. Start the API** (in one terminal):
```bash
uvicorn api:app --reload
```
Interactive docs available at `http://127.0.0.1:8000/docs`.

**2. Start the dashboard** (in a second terminal, with the API still running):
```bash
streamlit run dashboard.py
```
Opens at `http://localhost:8501`.

## Example

**Input:**
> "My device won't turn on no matter what I try. This is extremely urgent, I need help right now."

**Output:**
| Category | Confidence | Urgency | Confidence | Routing |
|---|---|---|---|---|
| Technical issue | 81.1% | Critical | 93.7% | ✅ Auto-routed |

## A note on the dataset

The models here are trained on a **synthetically generated dataset** (`synthetic_support_tickets.csv`), not real support tickets. An initial attempt used a public Kaggle dataset, but its category/priority labels turned out to be randomly assigned and uncorrelated with the ticket text (confirmed via a ~19% accuracy baseline, equivalent to random guessing across 5 classes). The synthetic dataset was built instead so the text-to-label mapping is genuine, letting the full pipeline be demonstrated end-to-end with meaningful evaluation metrics.

**Implication:** because the synthetic text is generated from a small set of templates, evaluation scores (F1 ≈ 1.00) are high due to limited linguistic variety, not because the task is trivial in general. On real-world tickets, expect meaningfully lower — but still useful — performance. Swapping in a real, correctly-labeled ticket dataset (keeping the same `text`, `category`, `urgency` column structure) would be the natural next step to validate this on production-like data.

## Possible next steps

- Replace synthetic data with a real or better-labeled ticket dataset
- Try sentence embeddings (e.g. `all-MiniLM-L6-v2`) instead of TF-IDF for better generalization
- Add authentication and persistent logging of low-confidence tickets to a review queue/database
- Deploy the API + dashboard (e.g. Render, Railway, or Docker + a cloud VM)

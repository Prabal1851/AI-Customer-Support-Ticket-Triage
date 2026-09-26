from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import csv
import os
from datetime import datetime

app = FastAPI()

category_model = joblib.load('models/category_model.pkl')
urgency_model = joblib.load('models/urgency_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

CONFIDENCE_THRESHOLD = 0.5  # below this, flag for human review

LOG_FILE = 'review_queue.csv'
LOG_FIELDS = ['timestamp', 'text', 'predicted_category', 'category_confidence',
              'predicted_urgency', 'urgency_confidence']

def log_for_review(text, cat_pred, cat_conf, urg_pred, urg_conf):
    """Append a low-confidence prediction to the review queue CSV, creating it with headers if needed."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'text': text,
            'predicted_category': cat_pred,
            'category_confidence': round(cat_conf, 3),
            'predicted_urgency': urg_pred,
            'urgency_confidence': round(urg_conf, 3),
        })

class Ticket(BaseModel):
    text: str

@app.post("/predict")
def predict(ticket: Ticket):
    X = vectorizer.transform([ticket.text])

    cat_probs = category_model.predict_proba(X)[0]
    cat_pred = category_model.classes_[cat_probs.argmax()]
    cat_conf = float(cat_probs.max())

    urg_probs = urgency_model.predict_proba(X)[0]
    urg_pred = urgency_model.classes_[urg_probs.argmax()]
    urg_conf = float(urg_probs.max())

    needs_review = cat_conf < CONFIDENCE_THRESHOLD or urg_conf < CONFIDENCE_THRESHOLD

    if needs_review:
        log_for_review(ticket.text, cat_pred, cat_conf, urg_pred, urg_conf)

    return {
        "category": cat_pred,
        "category_confidence": round(cat_conf, 3),
        "urgency": urg_pred,
        "urgency_confidence": round(urg_conf, 3),
        "needs_human_review": needs_review
    }

@app.get("/review-queue")
def get_review_queue():
    """Return all tickets currently logged for human review."""
    if not os.path.isfile(LOG_FILE):
        return {"count": 0, "tickets": []}

    with open(LOG_FILE, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return {"count": len(rows), "tickets": rows}

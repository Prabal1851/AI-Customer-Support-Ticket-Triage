from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

category_model = joblib.load('models/category_model.pkl')
urgency_model = joblib.load('models/urgency_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

CONFIDENCE_THRESHOLD = 0.5  # below this, flag for human review

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

    return {
        "category": cat_pred,
        "category_confidence": round(cat_conf, 3),
        "urgency": urg_pred,
        "urgency_confidence": round(urg_conf, 3),
        "needs_human_review": needs_review
    }
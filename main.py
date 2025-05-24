from fastapi import FastAPI
from schemas import LogMessage
import joblib

app = FastAPI(title="Log Classification API")

# Load model and vectorizer
vectorizer, model = joblib.load("model/classifier.pkl")

@app.post("/classify")
def classify_log(log: LogMessage):
    X = vectorizer.transform([log.message])
    prediction = model.predict(X)
    return {"log_type": prediction[0]}

"""Flask UI and JSON prediction endpoint."""
from pathlib import Path
import os
import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ROOT / "models" / "model.joblib")))
_model = None

def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError("Model missing. Run: python train_model.py")
        _model = joblib.load(MODEL_PATH)  # Only load files you created or trust.
    return _model

def predict_text(text):
    model = get_model()
    prediction = int(model.predict([text])[0])
    probability = float(model.predict_proba([text])[0][1])
    return {"prediction": "potentially_fraudulent" if prediction else "likely_genuine", "fraud_probability": round(probability, 4), "warning": "Educational estimate only; verify the employer independently."}

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.is_file()})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        return jsonify({"error": "Send JSON with a 'text' string"}), 400
    text = payload["text"].strip()
    if not (15 <= len(text) <= 10000):
        return jsonify({"error": "Text length must be between 15 and 10000 characters"}), 400
    try:
        return jsonify(predict_text(text))
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)

"""Train and compare baseline text classifiers. Run: python train_model.py"""
from pathlib import Path
import argparse
import csv
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent

def train(dataset, output):
    with open(dataset, encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    rows = [(r["text"].strip(), int(r["fraudulent"])) for r in records if r.get("text", "").strip()]
    if len(rows) < 20 or set(y for _, y in rows) != {0, 1}:
        raise ValueError("Dataset needs at least 20 labeled examples across both classes")
    X, y = zip(*rows)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    candidates = {"logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), "naive_bayes": MultinomialNB()}
    results, models = {}, {}
    for name, clf in candidates.items():
        pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        results[name] = {"accuracy": round(float(accuracy_score(y_test, pred)), 4), "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4), "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4), "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4), "confusion_matrix": confusion_matrix(y_test, pred).tolist()}
        models[name] = pipeline
    best = max(results, key=lambda n: results[n]["f1"])
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best], output / "model.joblib")
    metrics = {"selected_model": best, "train_samples": len(X_train), "test_samples": len(X_test), "evaluation": results, "dataset_note": "Small, synthetic demonstration dataset; scores do not represent real-world effectiveness."}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "demo_job_postings.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    train(args.data, args.output)

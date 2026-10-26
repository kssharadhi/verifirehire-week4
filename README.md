# VerifyHire — Week 4 AI Deployment Capstone

A Flask-based educational job-posting classifier using TF-IDF and scikit-learn. **The bundled data is synthetic and intentionally small. It is for demonstrating the ML pipeline, not real-world fraud detection.**

## Features
- Dataset loading and preprocessing through TF-IDF
- Train/test split (75/25, stratified)
- Compare Logistic Regression vs Multinomial Naive Bayes using F1 score
- Save the selected complete pipeline with Joblib
- Flask web UI and `POST /predict` JSON API
- Validation, health endpoint, basic pytest tests

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

## Example API
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"Pay a registration fee now to receive your job offer without an interview."}'
```
Returns a `prediction`, `fraud_probability`, and `warning`. The probability is a model estimate, **not** a verified real-world risk probability.

## Tests
```bash
python -m pytest -q
```

## Deploy (Render or similar Python hosting)
1. Push the project to GitHub.
2. Add `gunicorn>=21,<24` to requirements.txt if your hosting platform uses the provided Procfile (included already).
3. Create a Python web service with build command `pip install -r requirements.txt && python train_model.py`.
4. Set start command `gunicorn app:app` and use the public service URL.
5. Test `/health` and the web UI. Free hosting availability and plan terms may change.

## Using a real dataset
Replace `data/demo_job_postings.csv` with a legally usable, labeled CSV containing `text` and `fraudulent` (0 or 1). Train again and report actual held-out results. A representative dataset and careful leakage checks are essential before using this for real decisions.

## Safety and limitations
This model does not verify companies, recruiters, or payment requests. Synthetic examples produce misleadingly optimistic metrics. Job posts may be misclassified. Do not upload sensitive personal data. Only load trusted Joblib model files.

## Submission checklist
- [ ] Train and test on a representative dataset
- [ ] Capture genuine training metrics and screenshots
- [ ] Deploy and test public URL
- [ ] Add your GitHub and deployment URLs to report
- [ ] Submit DOCX and GitHub URL

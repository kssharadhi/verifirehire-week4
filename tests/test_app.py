from app import app

def test_home():
    assert app.test_client().get("/").status_code == 200

def test_invalid_json():
    response = app.test_client().post("/predict", json={"bad": "value"})
    assert response.status_code == 400

def test_short_text():
    response = app.test_client().post("/predict", json={"text": "short"})
    assert response.status_code == 400

def test_valid_prediction():
    response = app.test_client().post("/predict", json={"text": "Pay a registration fee immediately to get hired without interview."})
    assert response.status_code == 200
    assert 0 <= response.json["fraud_probability"] <= 1

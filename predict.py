import pickle

import xgboost as xgb
from fastapi import FastAPI

MODEL_FILE = 'model.bin'

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title='Diabetes readmission prediction')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(patient: dict):
    """Return the probability that a patient is readmitted within 30 days."""
    X = dv.transform([patient])
    probability = float(model.predict(xgb.DMatrix(X))[0])
    return {'readmission_probability': round(probability, 4)}

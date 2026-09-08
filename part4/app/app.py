"""Fraud-detector inference server.

Pre-instrumented with `prometheus-fastapi-instrumentator` — the /metrics
endpoint exposes request counters + latency histograms in Prometheus
text format.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

HERE = Path(__file__).resolve().parent
model = joblib.load(HERE / "model.pkl")

app = FastAPI(title="fraud-detector")
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(payload: dict):
    features = np.asarray(payload["features"], dtype=float).reshape(1, -1)
    pred = int(model.predict(features)[0])
    return {"prediction": pred}

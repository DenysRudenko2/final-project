"""FastAPI inference-сервіс з логуванням, Prometheus-метриками та drift-детектором."""
import logging
import os
import pickle
import time

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("aiops")

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl"))
DRIFT_Z = float(os.environ.get("DRIFT_Z_THRESHOLD", "3.0"))

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)
_model = bundle["model"]
_means = np.array(bundle["feature_means"])
_stds = np.array(bundle["feature_stds"])
_target_names = bundle["target_names"]
log.info(f"Model loaded from {MODEL_PATH}, accuracy={bundle.get('accuracy')}")

app = FastAPI(title="AIOps Quality Inference", version="1.0.0")

# Кастомні Prometheus-метрики (HTTP-метрики додає Instrumentator нижче)
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["predicted_class"])
DRIFT = Counter("drift_detected_total", "Number of drift detections")
LATENCY = Histogram("predict_latency_seconds", "Inference latency (s)")

# /metrics — http_requests_total, http_request_duration_seconds + кастомні метрики
Instrumentator().instrument(app).expose(app)


class Features(BaseModel):
    features: list[float]  # 4 фічі Iris: sepal/petal length/width


def detect_drift(features):
    """Простий статистичний drift-детектор: z-score фіч відносно тренувального розподілу."""
    z = np.abs((np.array(features, dtype=float) - _means) / _stds)
    return bool((z > DRIFT_Z).any()), float(z.max())


def predict(features):
    """Логіка інференсу — окрема функція (вимога ДЗ)."""
    x = np.array(features, dtype=float).reshape(1, -1)
    pred = int(_model.predict(x)[0])
    proba = float(_model.predict_proba(x)[0].max())
    return pred, proba


@app.post("/predict")
def predict_endpoint(payload: Features):
    start = time.time()
    log.info(f"request features={payload.features}")

    drifted, z_max = detect_drift(payload.features)
    if drifted:
        DRIFT.inc()
        log.warning(f"Drift detected: max_z={z_max:.2f} features={payload.features}")

    pred, proba = predict(payload.features)
    class_name = _target_names[pred]
    PREDICTIONS.labels(predicted_class=class_name).inc()
    LATENCY.observe(time.time() - start)

    resp = {
        "prediction": pred,
        "class_name": class_name,
        "probability": round(proba, 4),
        "drift_detected": drifted,
        "max_z_score": round(z_max, 3),
    }
    log.info(f"response {resp}")
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "aiops-quality-inference", "endpoints": ["/predict", "/metrics", "/health"]}


from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Prometheus metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the model is loaded"
)

model = None


def load_model():
    global model

    if model is not None:
        return model

    try:
        model = joblib.load(MODEL_PATH)
        MODEL_LOADED.set(1)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )

        return model

    except Exception:
        MODEL_LOADED.set(0)
        logger.exception("event=model_load_failed")
        raise


@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    duration_seconds = time.perf_counter() - g.start_time
    duration_ms = duration_seconds * 1000

    # Exclude Prometheus scrapes from user traffic
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s "
        "method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = g.request_id
    return response


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "message": "Practical 12 - Monitoring Enabled"
    })


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "event=validation_error request_id=%s "
            "reason=invalid_json",
            g.request_id
        )

        return jsonify({
            "error": "JSON object required"
        }), 400

    missing = [
        field for field in FEATURES
        if field not in data
    ]

    if missing:
        logger.warning(
            "event=validation_error request_id=%s "
            "missing_fields=%s",
            g.request_id,
            missing
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    logger.info(
        "event=prediction_request request_id=%s "
        "feature_names=%s",
        g.request_id,
        FEATURES
    )

    try:
        sample = pd.DataFrame([{
            field: data[field]
            for field in FEATURES
        }])

        for field in FEATURES:
            sample[field] = pd.to_numeric(
                sample[field],
                errors="raise"
            )

    except (ValueError, TypeError):
        logger.warning(
            "event=validation_error request_id=%s "
            "reason=invalid_numeric_input",
            g.request_id
        )

        return jsonify({
            "error": "All features must be numeric"
        }), 400

    try:
        trained_model = load_model()
        code = int(trained_model.predict(sample)[0])

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success request_id=%s "
            "prediction_code=%s",
            g.request_id,
            code
        )

        return jsonify({
            "prediction": "PASS" if code == 1 else "FAIL",
            "prediction_code": code
        })

    except Exception:
        PREDICTIONS.labels(result="error").inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


@app.errorhandler(404)
def route_not_found(error):
    logger.warning(
        "event=route_not_found request_id=%s path=%s",
        g.request_id,
        request.path
    )

    return jsonify({
        "error": "Route not found"
    }), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

# Configure logging for Render and Docker
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

# Cache the model after its first successful load
model = None


def load_model():
    global model

    if model is not None:
        return model

    try:
        model = joblib.load(MODEL_PATH)
        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )
        return model

    except Exception:
        logger.exception("event=model_load_failed")
        raise


# Log the beginning of each HTTP request
@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.perf_counter()


# Log response status and duration
@app.after_request
def after_request(response):
    duration_ms = (
        time.perf_counter() - g.start_time
    ) * 1000

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


# Root / health endpoint
@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "message": "Practical 11 - Logging Enabled"
    })


# Prediction endpoint
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

        # Reject invalid numeric input
        for field in FEATURES:
            sample[field] = pd.to_numeric(
                sample[field], errors="raise"
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
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# Log requests to nonexistent endpoints
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

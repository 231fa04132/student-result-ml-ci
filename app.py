
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s "
           "request_id=%(request_id)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())

logger = logging.getLogger(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("model_loaded")
        return model
    except Exception:
        logger.exception("model_load_failed")
        raise


@app.before_request
def start_request():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    request_id = getattr(g, "request_id", "-")
    duration_ms = (
        (time.perf_counter() - g.start_time) * 1000
        if hasattr(g, "start_time") else 0
    )

    logger.info(
        "http_request method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.path,
        response.status_code,
        duration_ms,
        extra={"request_id": request_id},
    )

    response.headers["X-Request-ID"] = request_id
    return response


@app.errorhandler(Exception)
def handle_exception(error):
    if isinstance(error, HTTPException):
        return jsonify({"error": error.description}), error.code

    logger.exception(
        "unhandled_error",
        extra={"request_id": getattr(g, "request_id", "-")},
    )
    return jsonify({"error": "Internal server error"}), 500


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning("invalid_json_request")
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "missing_prediction_fields fields=%s", missing_fields
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        logger.info("prediction_started")

        model = load_model()
        prediction_code = int(model.predict(sample)[0])
        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "prediction_succeeded prediction=%s code=%s",
            prediction,
            prediction_code,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        })

    except (ValueError, TypeError) as error:
        logger.warning("invalid_prediction_input error=%s", error)
        return jsonify({"error": "Invalid feature values"}), 400

    except Exception:
        logger.exception("prediction_failed")
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

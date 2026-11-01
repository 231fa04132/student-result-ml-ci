
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

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score",
]

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")


# Load and log the model
def load_model():
    if not MODEL_PATH.exists():
        logger.error(
            "event=model_load_failed model_file=%s",
            MODEL_PATH,
        )
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    model = joblib.load(MODEL_PATH)
    logger.info(
        "event=model_loaded model_file=%s",
        MODEL_PATH,
    )
    return model


# Request logging middleware equivalent for Flask
@app.before_request
def log_request_start():
    g.request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def log_request_end(response):
    request_id = getattr(
        g, "request_id", str(uuid.uuid4())
    )

    start_time = getattr(
        g, "start_time", time.perf_counter()
    )

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    response.headers["X-Request-ID"] = request_id

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s "
        "method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


# Handle unexpected application errors
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error

    request_id = getattr(g, "request_id", "unknown")

    logger.exception(
        "event=application_error request_id=%s "
        "error_type=%s",
        request_id,
        type(error).__name__,
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": request_id,
    }), 500


# Health check
@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
    })


# Student result prediction
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "event=invalid_json path=/predict"
        )
        return jsonify({
            "error": "A valid JSON object is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error path=/predict "
            "missing_fields=%s",
            missing_fields,
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        # Validate numeric inputs without logging their values
        sample_data = {}

        for feature in FEATURES:
            value = data[feature]

            if isinstance(value, bool):
                raise ValueError(
                    f"{feature} must be numeric"
                )

            sample_data[feature] = float(value)

        sample = pd.DataFrame(
            [sample_data],
            columns=FEATURES,
        )

        logger.info(
            "event=prediction_request features=%s",
            FEATURES,
        )

        model = load_model()

        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        logger.info(
            "event=prediction_success prediction=%s",
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
        })

    except (ValueError, TypeError) as error:
        logger.warning(
            "event=validation_error error_type=%s",
            type(error).__name__,
        )

        return jsonify({
            "error": "Input features must be numeric"
        }), 400

    except Exception:
        logger.exception(
            "event=prediction_failed"
        )
        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    logger.info("event=application_starting")
    app.run(host="0.0.0.0", port=5000)

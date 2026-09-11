"""
Predict whether one laughter clip is real or fake.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from feature_extraction import extract_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"

LABEL_NAMES = {0: "REAL", 1: "FAKE"}


def load_model():
    model_path = MODEL_DIR / "laugh_classifier.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python src/train_model.py` first."
        )

    return joblib.load(model_path), joblib.load(scaler_path)


def predict(file_path: str | Path) -> tuple[str, float | None]:
    classifier, scaler = load_model()

    features = extract_features(file_path).reshape(1, -1)
    scaled_features = scaler.transform(features)

    predicted_label = int(classifier.predict(scaled_features)[0])
    confidence = None

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(scaled_features)[0]
        confidence = float(probabilities[predicted_label])

    return LABEL_NAMES[predicted_label], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to an audio clip.")
    args = parser.parse_args()

    try:
        label, confidence = predict(args.file)

        print(f"Prediction: {label}")
        if confidence is not None:
            print(f"Confidence: {confidence * 100:.1f}%")
    except Exception as error:
        print(f"ERROR: {error}")
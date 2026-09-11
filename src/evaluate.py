"""
Evaluate the saved fake-laugh classifier and generate plots.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from feature_extraction import FEATURE_NAMES, extract_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
PLOT_DIR = MODEL_DIR / "plots"

LABELS = {"real": 0, "fake": 1}


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []

    for class_name, class_label in LABELS.items():
        for audio_file in sorted((DATA_DIR / class_name).glob("*.wav")):
            try:
                features.append(extract_features(audio_file))
                labels.append(class_label)
            except Exception as error:
                print(f"Skipped {audio_file.name}: {error}")

    return np.asarray(features), np.asarray(labels)


def main() -> None:
    model_path = MODEL_DIR / "laugh_classifier.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"

    if not model_path.exists() or not scaler_path.exists():
        print("ERROR: Train the model first using: python src/train_model.py")
        return

    X, y = load_dataset()

    if len(X) == 0 or len(np.unique(y)) < 2:
        print("ERROR: Evaluation requires audio clips in both classes.")
        return

    classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_test_scaled = scaler.transform(X_test)
    predictions = classifier.predict(X_test_scaled)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y_test, predictions, target_names=["REAL", "FAKE"]))

    metrics = {
        "accuracy": float(accuracy),
        "test_samples": int(len(y_test)),
    }

    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Fake Laugh Detector — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    if hasattr(classifier, "predict_proba") and len(np.unique(y_test)) == 2:
        probabilities = classifier.predict_proba(X_test_scaled)[:, 1]
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            probabilities,
        )
        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.figure(figsize=(5, 4))
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"AUC = {roc_auc:.2f}",
        )
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150)
        plt.close()

    if hasattr(classifier, "feature_importances_"):
        indexes = np.argsort(classifier.feature_importances_)[::-1][:15]

        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=classifier.feature_importances_[indexes],
            y=[FEATURE_NAMES[index] for index in indexes],
        )
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "feature_importance.png", dpi=150)
        plt.close()

    print(f"\nPlots saved in: {PLOT_DIR}")


if __name__ == "__main__":
    main()
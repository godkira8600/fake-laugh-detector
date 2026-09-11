"""
Train a classifier to distinguish genuine and posed laughter.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from feature_extraction import FEATURE_NAMES, extract_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

LABELS = {"real": 0, "fake": 1}
LABEL_NAMES = {0: "REAL", 1: "FAKE"}
MIN_SAMPLES_PER_CLASS = 5


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []

    for class_name, class_label in LABELS.items():
        folder = DATA_DIR / class_name
        audio_files = sorted(folder.glob("*.wav"))

        print(f"\n{class_name.upper()}: {len(audio_files)} file(s) found")

        for audio_file in audio_files:
            try:
                features.append(extract_features(audio_file))
                labels.append(class_label)
                print(f"  Processed: {audio_file.name}")
            except Exception as error:
                print(f"  Skipped {audio_file.name}: {error}")

    return np.asarray(features), np.asarray(labels)


def build_classifier(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        )

    return SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
        class_weight="balanced",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["rf", "svm"],
        default="rf",
        help="Classifier to use: rf or svm.",
    )
    args = parser.parse_args()

    print("Loading dataset and extracting features...")
    X, y = load_dataset()

    if len(X) == 0:
        print("\nERROR: No usable audio clips were found.")
        return

    counts = {name: int(np.sum(y == label)) for name, label in LABELS.items()}
    print(f"\nDataset summary: {counts}")

    if min(counts.values()) < MIN_SAMPLES_PER_CLASS:
        print(
            f"\nERROR: Add at least {MIN_SAMPLES_PER_CLASS} usable WAV clips "
            "to both data/real and data/fake before training."
        )
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = build_classifier(args.model)
    classifier.fit(X_train_scaled, y_train)

    predictions = classifier.predict(X_test_scaled)

    print("\nHeld-out test performance:")
    print(classification_report(y_test, predictions, target_names=["REAL", "FAKE"]))

    min_class_count = min(counts.values())
    folds = min(5, min_class_count)

    cross_validation_model = make_pipeline(
        StandardScaler(),
        build_classifier(args.model),
    )
    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(cross_validation_model, X, y, cv=cross_validation)
    print(
        f"Cross-validation accuracy: {cv_scores.mean():.3f} "
        f"(+/- {cv_scores.std():.3f})"
    )

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(classifier, MODEL_DIR / "laugh_classifier.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    print(f"\nSaved model: {MODEL_DIR / 'laugh_classifier.joblib'}")
    print(f"Saved scaler: {MODEL_DIR / 'scaler.joblib'}")

    if hasattr(classifier, "feature_importances_"):
        print("\nTop 10 important features:")
        top_indexes = np.argsort(classifier.feature_importances_)[::-1][:10]

        for index in top_indexes:
            print(
                f"  {FEATURE_NAMES[index]:30} "
                f"{classifier.feature_importances_[index]:.4f}"
            )


if __name__ == "__main__":
    main()
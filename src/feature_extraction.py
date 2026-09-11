"""
Extract a fixed-length acoustic feature vector from a laughter audio clip.
"""

from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050
N_MFCC = 13

FEATURE_NAMES = (
    [f"mfcc_{index + 1}_mean" for index in range(N_MFCC)]
    + [f"mfcc_{index + 1}_std" for index in range(N_MFCC)]
    + ["pitch_mean", "pitch_std", "pitch_min", "pitch_max"]
    + ["rms_mean", "rms_std"]
    + ["zcr_mean", "zcr_std"]
    + [
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
    ]
    + ["tempo"]
)


def _pitch_statistics(y: np.ndarray, sr: int) -> list[float]:
    """Return stable F0 statistics even when pitch cannot be detected."""
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )

        if f0 is None or voiced_flag is None:
            return [0.0, 0.0, 0.0, 0.0]

        voiced_f0 = f0[voiced_flag]
        voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

        if len(voiced_f0) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        return [
            float(np.mean(voiced_f0)),
            float(np.std(voiced_f0)),
            float(np.min(voiced_f0)),
            float(np.max(voiced_f0)),
        ]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def extract_features(file_path: str | Path) -> np.ndarray:
    """
    Extract 38 acoustic features from one audio file.

    Returns:
        NumPy array with shape (38,).
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    if len(y) == 0:
        raise ValueError("The audio file contains no samples.")

    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) < int(sr * 0.2):
        y = np.pad(y, (0, int(sr * 0.2) - len(y)))

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    pitch_stats = _pitch_statistics(y, sr)

    rms = librosa.feature.rms(y=y)[0]
    rms_stats = [float(rms.mean()), float(rms.std())]

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_stats = [float(zcr.mean()), float(zcr.std())]

    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean())
    bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean())
    rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean())

    try:
        onset_envelope = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=sr,
        )
        tempo = float(np.asarray(tempo).flatten()[0])
    except Exception:
        tempo = 0.0

    features = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            pitch_stats,
            rms_stats,
            zcr_stats,
            [centroid, bandwidth, rolloff, tempo],
        ]
    )

    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError("Feature extraction returned an unexpected feature count.")

    return features.astype(np.float32)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/feature_extraction.py <audio_file>")
        raise SystemExit(1)

    extracted_features = extract_features(sys.argv[1])

    for name, value in zip(FEATURE_NAMES, extracted_features):
        print(f"{name:30}: {value:.4f}")
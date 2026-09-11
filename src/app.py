import os
import random
import tempfile
from pathlib import Path

import librosa
import streamlit as st

from predict import predict

PROJECT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_DIR / "assets"
FAKE_LAUGH_SOUND = ASSETS_DIR / "sfx" / "fake_laugh.wav"
ACHIEVEMENT_DIR = ASSETS_DIR / "achievements"

MAX_DURATION_SECONDS = 10

ACHIEVEMENTS = [
    {
        "title": "ചിരി ചാമ്പ്യൻ",
        "message": "മികച്ച ചിരി പ്രകടനത്തിന് അഭിനന്ദനങ്ങൾ.",
        "audio": ACHIEVEMENT_DIR / "chirippu_champion.wav",
    },
    {
        "title": "ചിരിയുടെ രാജാവ് / രാജ്ഞി",
        "message": "നിങ്ങളുടെ ചിരി ശ്രദ്ധേയമാണ്.",
        "audio": ACHIEVEMENT_DIR / "chiriyude_raja.wav",
    },
    {
        "title": "കിലുക്കം ഹീറോ",
        "message": "നല്ല ഊർജ്ജമുള്ള ചിരി കണ്ടെത്തി.",
        "audio": ACHIEVEMENT_DIR / "kilukkam_hero.wav",
    },
    {
        "title": "കോമഡി ഡിറ്റക്ടീവ്",
        "message": "ചിരിയുടെ പ്രത്യേകത കണ്ടെത്തിയിരിക്കുന്നു.",
        "audio": ACHIEVEMENT_DIR / "comedy_detective.wav",
    },
]


def get_mime_type(audio_path: Path) -> str:
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
    }
    return mime_types.get(audio_path.suffix.lower(), "audio/wav")


def get_audio_duration(audio_path: str) -> float:
    return float(librosa.get_duration(path=audio_path))


def play_asset(audio_path: Path, autoplay: bool = False) -> None:
    if audio_path.exists():
        st.audio(
            str(audio_path),
            format=get_mime_type(audio_path),
            autoplay=autoplay,
        )


st.set_page_config(
    page_title="Laugh Analysis Studio",
    page_icon="◉",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: #f6f8fb;
            color: #172033;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 248, 251, 0.92);
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            background: linear-gradient(135deg, #172554 0%, #1e3a8a 100%);
            border-radius: 18px;
            padding: 2rem;
            color: #ffffff;
            box-shadow: 0 10px 28px rgba(30, 58, 138, 0.18);
            animation: slideDown 0.55s ease-out;
        }

        .app-header h1 {
            margin: 0;
            color: #ffffff;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.6px;
        }

        .app-header p {
            margin: 0.55rem 0 0;
            color: #dbeafe;
            font-size: 1rem;
        }

        .section-title {
            color: #172554;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 1.4rem 0 0.45rem;
        }

        .info-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 1rem 1.15rem;
            color: #334155;
            box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
        }

        .result-card {
            margin-top: 1rem;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 16px;
            border: 1px solid #dbe3ef;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.09);
            animation: revealResult 0.5s ease-out;
        }

        .real-result {
            border-left: 6px solid #16a34a;
        }

        .fake-result {
            border-left: 6px solid #dc2626;
        }

        .result-card h2 {
            color: #172554;
            margin: 0 0 0.45rem;
            font-size: 1.35rem;
        }

        .result-card p {
            color: #475569;
            margin: 0;
        }

        .badge-card {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
            color: #1e3a8a;
        }

        .stButton > button {
            width: 100%;
            border-radius: 9px;
            background: #1d4ed8;
            color: #ffffff;
            border: 1px solid #1d4ed8;
            padding: 0.65rem;
            font-weight: 600;
            transition: all 0.18s ease;
        }

        .stButton > button:hover {
            background: #1e40af;
            border-color: #1e40af;
            color: #ffffff;
            transform: translateY(-1px);
        }

        button[data-baseweb="tab"] {
            font-weight: 600;
            color: #475569;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #1d4ed8;
        }

        @keyframes slideDown {
            from { opacity: 0; transform: translateY(-12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes revealResult {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        [data-testid="stMetricLabel"] {
            color: #000000 !important;
        }

        [data-testid="stMetricValue"] {
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>Laugh Analysis Studio</h1>
        <p>
            Audio-based analysis for distinguishing spontaneous and posed laughter.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_right = st.columns([2, 1])

with top_left:
    st.markdown(
        """
        <div class="section-title">Audio input</div>
        <div class="info-card">
            Record a short clip or choose an existing audio file.
            Each recording must be 10 seconds or less.
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.metric("Maximum duration", "10 seconds")

record_tab, upload_tab = st.tabs(["Record audio", "Upload file"])

recorded_audio = None
uploaded_audio = None

with record_tab:
    recorded_audio = st.audio_input(
        "Record a laughter clip (maximum 10 seconds)",
    )

    if recorded_audio is not None:
        st.success("Recording captured successfully.")
        st.audio(recorded_audio)

        st.download_button(
            label="Download WAV recording",
            data=recorded_audio.getvalue(),
            file_name="laughter_recording.wav",
            mime="audio/wav",
        )

with upload_tab:
    uploaded_audio = st.file_uploader(
        "Choose WAV, MP3, or FLAC audio",
        type=["wav", "mp3", "flac"],
    )

    if uploaded_audio is not None:
        st.success(f"Selected file: {uploaded_audio.name}")
        st.audio(uploaded_audio)

if recorded_audio is not None and uploaded_audio is not None:
    st.info("The microphone recording will be analyzed.")

audio_source = recorded_audio if recorded_audio is not None else uploaded_audio

if audio_source is not None:
    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)

    if st.button("Run laugh analysis"):
        temporary_path = None

        try:
            suffix = ".wav"

            if recorded_audio is None:
                suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
                file.write(audio_source.getvalue())
                temporary_path = file.name

            duration = get_audio_duration(temporary_path)

            if duration > MAX_DURATION_SECONDS:
                st.error(
                    f"Audio duration is {duration:.1f} seconds. "
                    f"Please use an audio clip of {MAX_DURATION_SECONDS} seconds or less."
                )
            else:
                with st.status(
                    "Processing audio signal...",
                    expanded=True,
                ) as status:
                    st.write("Reading audio waveform")
                    st.write("Extracting acoustic features")
                    st.write("Running classifier inference")

                    label, confidence = predict(temporary_path)

                    status.update(
                        label="Analysis complete",
                        state="complete",
                        expanded=False,
                    )

                achievement = random.choice(ACHIEVEMENTS)

                result_left, result_right = st.columns([2, 1])

                with result_left:
                    if label == "REAL":
                        st.balloons()

                        st.markdown(
                            """
                            <div class="result-card real-result">
                                <h2>Classification: Genuine / Real laughter</h2>
                                <p>
                                    The model found acoustic patterns more consistent
                                    with spontaneous laughter.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="result-card fake-result">
                                <h2>Classification: Posed / Fake laughter</h2>
                                <p>
                                    The model found acoustic patterns more consistent
                                    with posed or acted laughter.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if FAKE_LAUGH_SOUND.exists():
                            st.caption("Optional response audio")
                            play_asset(FAKE_LAUGH_SOUND, autoplay=True)

                with result_right:
                    if confidence is not None:
                        st.metric(
                            "Model confidence",
                            f"{confidence * 100:.1f}%",
                        )
                        st.progress(min(max(confidence, 0.0), 1.0))

                st.markdown(
                    f"""
                    <div class="badge-card">
                        <b>Achievement: {achievement["title"]}</b><br>
                        {achievement["message"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if achievement["audio"].exists():
                    st.caption("Malayalam achievement audio")
                    play_asset(achievement["audio"])
                else:
                    st.caption("Achievement audio file has not been added yet.")

        except FileNotFoundError:
            st.warning(
                "No trained model was found. "
                "Train the model before running analysis."
            )

        except Exception as error:
            st.error(f"Unable to process this audio file: {error}")

        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

else:
    st.markdown(
        """
        <div class="info-card" style="margin-top: 1.4rem;">
            Select an input method above to begin audio analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.caption(
    "This project is an educational audio-classification system. "
    "Its output should not be treated as a reliable assessment of intent or honesty."
)
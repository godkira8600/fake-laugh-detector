import os
import random
import tempfile
from pathlib import Path

import librosa
import streamlit as st
from predict import predict

# ──────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Laugh Analysis Studio",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# Dark theme + layout styles
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Base */
        .stApp {
            background: #0b0f19;
            color: #e2e8f0;
        }
        [data-testid="stHeader"] {
            background: rgba(11, 15, 25, 0.9);
            backdrop-filter: blur(8px);
        }
        .main .block-container {
            max-width: 1080px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        #MainMenu, footer {visibility: hidden;}

        /* Background decorative icons */
        .bg-icons {
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }
        .bg-icon {
            position: absolute;
            opacity: 0.06;
            user-select: none;
            filter: grayscale(40%);
        }

        /* Hero */
        .hero {
            background: linear-gradient(135deg, #111827 0%, #1e293b 55%, #0f172a 100%);
            border: 1px solid #1e293b;
            border-radius: 18px;
            padding: 2rem 2.3rem;
            margin-bottom: 1.6rem;
            position: relative;
            z-index: 1;
            box-shadow: 0 12px 36px rgba(0,0,0,0.4);
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.5px;
        }
        .hero p {
            margin: 0.5rem 0 0;
            color: #94a3b8;
            font-size: 1rem;
            max-width: 520px;
            line-height: 1.5;
        }
        .hero-badge {
            display: inline-block;
            margin-top: 1rem;
            background: rgba(59,130,246,0.15);
            border: 1px solid rgba(59,130,246,0.3);
            border-radius: 999px;
            padding: 0.28rem 0.9rem;
            font-size: 0.78rem;
            color: #93c5fd;
        }

        /* Section label */
        .section-label {
            font-size: 0.72rem;
            font-weight: 650;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #64748b;
            margin: 0 0 0.6rem;
            position: relative;
            z-index: 1;
        }

        /* Cards */
        .card {
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1.2rem;
            position: relative;
            z-index: 1;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .card-header {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 1rem;
            font-weight: 600;
            font-size: 1.05rem;
            color: #e2e8f0;
        }
        .card-soft {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
            color: #94a3b8;
            font-size: 0.95rem;
        }

        /* Result cards */
        .result-card {
            padding: 1.4rem 1.5rem;
            background: #111827;
            border-radius: 14px;
            border: 1px solid #1e293b;
            margin-top: 0.8rem;
            position: relative;
            z-index: 1;
        }
        .real-result { border-left: 5px solid #22c55e; }
        .fake-result { border-left: 5px solid #ef4444; }
        .result-card h2 {
            margin: 0 0 0.35rem;
            font-size: 1.25rem;
            color: #f1f5f9;
        }
        .result-card p {
            margin: 0;
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.5;
        }

        /* Achievement */
        .badge-card {
            background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            border: 1px solid #1e40af;
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            margin-top: 1rem;
            color: #93c5fd;
            position: relative;
            z-index: 1;
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            background: #2563eb !important;
            color: #fff !important;
            border: 1px solid #2563eb !important;
            padding: 0.7rem;
            font-weight: 600;
            box-shadow: 0 4px 16px rgba(37,99,235,0.35);
        }
        .stButton > button:hover {
            background: #1d4ed8 !important;
            border-color: #1d4ed8 !important;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            font-weight: 600;
            color: #64748b !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #60a5fa !important;
        }
        [data-baseweb="tab-highlight"] {
            background-color: #3b82f6 !important;
        }

        /* Metrics */
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
        [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

        /* Progress */
        .stProgress > div > div { background-color: #3b82f6 !important; }

        /* Footer */
        .footer-note {
            text-align: center;
            color: #475569;
            font-size: 0.8rem;
            line-height: 1.5;
            margin-top: 1.5rem;
            position: relative;
            z-index: 1;
        }

        /* Floating result emojis */
        .float-container {
            position: fixed;
            inset: 0;
            pointer-events: none;
            overflow: hidden;
            z-index: 9999;
        }
        .float-item {
            position: absolute;
            bottom: -50px;
            opacity: 0;
            animation-name: floatUp;
            animation-timing-function: ease-out;
            animation-fill-mode: forwards;
            user-select: none;
        }
        @keyframes floatUp {
            0%   { transform: translateY(0) rotate(0deg) scale(0.6); opacity: 0; }
            12%  { opacity: 1; }
            90%  { opacity: 0.8; }
            100% { transform: translateY(-110vh) rotate(680deg) scale(1.05); opacity: 0; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Background decorative icons (mics & speakers)
# ──────────────────────────────────────────────
icons = ["🎤", "🎙", "🔊", "🔈", "🎧", "📻", "📢"]
positions = [
    (5, 6), (10, 48), (7, 82), (20, 15), (25, 68),
    (35, 4), (42, 38), (48, 75), (58, 18), (65, 55),
    (72, 8), (80, 42), (88, 70), (15, 30), (55, 12),
    (30, 85), (70, 32), (92, 22), (3, 55), (40, 90),
]

bg_html = '<div class="bg-icons">'
for i, (top, left) in enumerate(positions):
    size = random.randint(26, 48)
    rotate = random.randint(-20, 20)
    icon = icons[i % len(icons)]
    bg_html += (
        f'<div class="bg-icon" style="top:{top}%;left:{left}%;'
        f'font-size:{size}px;transform:rotate({rotate}deg);">{icon}</div>'
    )
bg_html += "</div>"
st.markdown(bg_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Hero section
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Laugh Analysis Studio</h1>
        <p>
            Professional audio analysis that distinguishes spontaneous laughter
            from posed or acted laughter using acoustic features.
        </p>
        <div class="hero-badge">Educational · Audio Classification · ≤ 10 s clips</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Studio Console (Audio Input)
# ──────────────────────────────────────────────
st.markdown('<div class="section-label">Studio Console</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-header"><span>🎙</span> Audio Input</div>',
        unsafe_allow_html=True,
    )

    col_info, col_metric = st.columns([3, 1], gap="medium")
    with col_info:
        st.markdown(
            """
            <div class="card-soft">
                Record a short clip or upload an existing file.
                Keep the duration at <b style="color:#93c5fd;">10 seconds or less</b>
                for best results.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_metric:
        st.metric("Max duration", "10 s")

    st.write("")

    record_tab, upload_tab = st.tabs(["🎙  Record audio", "📁  Upload file"])

    recorded_audio = None
    uploaded_audio = None

    with record_tab:
        recorded_audio = st.audio_input(
            "Speak or laugh into the microphone",
            help="Maximum recommended length: 10 seconds",
        )
        if recorded_audio is not None:
            st.success("Recording captured")
            st.audio(recorded_audio)
            st.download_button(
                label="Download recording (WAV)",
                data=recorded_audio.getvalue(),
                file_name="laughter_recording.wav",
                mime="audio/wav",
                use_container_width=True,
            )

    with upload_tab:
        uploaded_audio = st.file_uploader(
            "Choose a WAV, MP3 or FLAC file",
            type=["wav", "mp3", "flac"],
            help="Files longer than 10 seconds will be rejected",
        )
        if uploaded_audio is not None:
            st.success(f"Selected · {uploaded_audio.name}")
            st.audio(uploaded_audio)

    st.markdown("</div>", unsafe_allow_html=True)

# Priority rule
if recorded_audio is not None and uploaded_audio is not None:
    st.info("Microphone recording will be used for analysis.")

audio_source = recorded_audio if recorded_audio is not None else uploaded_audio

# ──────────────────────────────────────────────
# Analysis section
# ──────────────────────────────────────────────
if audio_source is not None:
    st.markdown(
        '<div class="section-label" style="margin-top:1.4rem;">Analysis</div>',
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header"><span>🔬</span> Run Classification</div>',
            unsafe_allow_html=True,
        )

        if st.button("Run laugh analysis", type="primary", use_container_width=True):
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
                        f"Audio is {duration:.1f} s long. "
                        f"Please provide a clip of {MAX_DURATION_SECONDS} seconds or less."
                    )
                else:
                    with st.status("Processing audio signal…", expanded=True) as status:
                        st.write("Reading waveform")
                        st.write("Extracting acoustic features")
                        st.write("Running classifier")
                        label, confidence = predict(temporary_path)
                        status.update(
                            label="Analysis complete",
                            state="complete",
                            expanded=False,
                        )

                    achievement = random.choice(ACHIEVEMENTS)

                    # Floating emojis
                    if label == "REAL":
                        floating = ["😂", "🤣", "😆", "🥳", "🎉", "✨", "🤪", "😹", "💥", "🌟"]
                    else:
                        floating = ["😠", "😡", "🤬", "💢", "😤", "👿", "🔥", "👎", "😾", "💥"]

                    emoji_html = '<div class="float-container">'
                    for emoji in floating * 3:
                        left = random.randint(3, 94)
                        delay = round(random.uniform(0, 4.2), 2)
                        dur = round(random.uniform(4.8, 8.2), 2)
                        size = random.randint(22, 40)
                        emoji_html += (
                            f'<span class="float-item" style="left:{left}%;'
                            f'animation-delay:{delay}s;animation-duration:{dur}s;'
                            f'font-size:{size}px;">{emoji}</span>'
                        )
                    emoji_html += "</div>"
                    st.markdown(emoji_html, unsafe_allow_html=True)

                    # Results
                    left_col, right_col = st.columns([2.2, 1], gap="large")

                    with left_col:
                        if label == "REAL":
                            st.balloons()
                            st.markdown(
                                """
                                <div class="result-card real-result">
                                    <h2>Genuine / Real laughter</h2>
                                    <p>
                                        Acoustic patterns are consistent with spontaneous,
                                        unforced laughter.
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div class="result-card fake-result">
                                    <h2>Posed / Fake laughter</h2>
                                    <p>
                                        Acoustic patterns are more consistent with
                                        posed or acted laughter.
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if FAKE_LAUGH_SOUND.exists():
                                st.caption("Optional response audio")
                                play_asset(FAKE_LAUGH_SOUND, autoplay=True)

                    with right_col:
                        if confidence is not None:
                            st.metric("Model confidence", f"{confidence * 100:.1f}%")
                            st.progress(min(max(confidence, 0.0), 1.0))

                    st.markdown(
                        f"""
                        <div class="badge-card">
                            <b>Achievement · {achievement["title"]}</b><br>
                            <span style="opacity:0.9">{achievement["message"]}</span>
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
                st.warning("No trained model found. Train the model before running analysis.")
            except Exception as error:
                st.error(f"Unable to process this audio file: {error}")
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; color:#64748b; margin-top:1.4rem;">
            <div style="font-size:1.7rem; margin-bottom:0.4rem;">🎙</div>
            Choose a recording or upload a file above to begin analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-note">
        Educational audio-classification system.<br>
        Output should not be treated as a reliable assessment of intent or honesty.
    </div>
    """,
    unsafe_allow_html=True,
)
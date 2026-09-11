"""
Streamlit web app for the Fake Laugh Detector.
"""

import os
import tempfile

import streamlit as st

from predict import predict

st.set_page_config(page_title="Fake Laugh Detector", page_icon="😂")

st.title("😂 Fake Laugh Detector")
st.write(
    "Record or upload a laughter clip. The model predicts whether it is "
    "**genuine (REAL)** or **posed (FAKE)**."
)

st.subheader("Option 1: Record from microphone")

recorded_audio = st.audio_input(
    "Press the microphone button, record your laughter, then stop recording."
)

st.subheader("Option 2: Upload an audio file")

uploaded_audio = st.file_uploader(
    "Upload WAV, MP3, or FLAC audio",
    type=["wav", "mp3", "flac"],
)

audio_source = recorded_audio or uploaded_audio

if recorded_audio is not None:
    st.success("Recording captured as a WAV file.")
    st.audio(recorded_audio)

    st.download_button(
        label="Download recording as WAV",
        data=recorded_audio.getvalue(),
        file_name="recorded_laugh.wav",
        mime="audio/wav",
    )

if uploaded_audio is not None:
    st.audio(uploaded_audio)

if audio_source is not None:
    if st.button("Analyze laughter", type="primary"):
        temporary_path = None

        try:
            if recorded_audio is not None:
                suffix = ".wav"
            else:
                suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
                file.write(audio_source.getvalue())
                temporary_path = file.name

            with st.spinner("Analyzing laughter..."):
                label, confidence = predict(temporary_path)

            if label == "REAL":
                st.success("Prediction: REAL / genuine laughter")
            else:
                st.error("Prediction: FAKE / posed laughter")

            if confidence is not None:
                st.metric("Model confidence", f"{confidence * 100:.1f}%")
                st.progress(min(max(confidence, 0.0), 1.0))

        except FileNotFoundError as error:
            st.warning(str(error))

        except Exception as error:
            st.error(f"Could not analyze this audio: {error}")

        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

st.divider()
st.caption(
    "The microphone recording is WAV format. The model uses MFCC, pitch, "
    "energy, spectral, and rhythm features."
)
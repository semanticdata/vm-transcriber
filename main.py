import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import sys

if sys.platform.startswith("win"):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import whisper
import tempfile

st.set_page_config(page_title="Speech Transcription Demo")
st.title("🎤 Speech Transcription Demo")

st.write(
    "Upload an audio file (wav, mp3, m4a, etc.), play it, and transcribe using OpenAI Whisper."
)

uploaded_file = st.file_uploader(
    "Choose an audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"]
)

# Whisper model options
model_options = ["tiny", "base", "small", "medium", "large"]
model_choice = st.selectbox("Select Whisper model", model_options, index=1)

# Optional: Language hint
language_hint = st.text_input(
    "Language hint (optional, e.g. 'en', 'es', 'fr')", value=""
)

if uploaded_file is not None:
    # Play the audio
    st.audio(uploaded_file, format=uploaded_file.type)

    # Save to a temp file for whisper
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=uploaded_file.name[-4:]
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Manual transcription trigger
    if st.button("Transcribe"):
        # Show a temporary notification using st.toast if available (Streamlit >=1.29)
        try:
            st.toast(
                f"Transcribing with model '{model_choice}'... This may take a while."
            )
        except AttributeError:
            st.info(
                f"Transcribing with model '{model_choice}'... This may take a while."
            )
        try:
            model = whisper.load_model(model_choice)
            result = model.transcribe(tmp_path)
            try:
                st.toast("Transcription complete!")
            except AttributeError:
                st.success("Transcription complete!")
            st.text_area("Transcription", result["text"], height=200)
        except Exception as e:
            st.error(f"Error during transcription: {e}")

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

# Compact input header
col1, col2 = st.columns([2, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        accept_multiple_files=False,
    )
with col2:
    model_options = ["tiny", "base", "small", "medium", "large"]
    model_choice = st.selectbox("Model", model_options, index=1)
    LANGUAGES = {"English": "en", "Spanish": "es", "French": "fr"}
    language_display = st.selectbox("Language", list(LANGUAGES.keys()), index=0)
    language_hint = LANGUAGES[language_display]

st.write(
    ":arrow_up: Upload an audio file, select your model, and transcribe using OpenAI Whisper."
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

    # Helper to save annotated entry
    @st.cache_data(show_spinner=False)
    def save_entry(entry):
        if "entries" not in st.session_state:
            st.session_state["entries"] = []
        st.session_state["entries"].append(entry)
        return st.session_state["entries"]

    def entry_to_txt(entry):
        return f"Name: {entry['name']}\nPhone: {entry['phone']}\nAddress: {entry['address']}\nNotes: {entry['notes']}\n---\nTranscription:\n{entry['transcription']}\n"

    # Cache the Whisper model per model_choice
    @st.cache_resource(show_spinner=False)
    def get_whisper_model(model_choice):
        return whisper.load_model(model_choice)

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
            model = get_whisper_model(model_choice)
            result = model.transcribe(tmp_path)
            try:
                st.toast("Transcription complete!")
            except AttributeError:
                st.success("Transcription complete!")
            # Store transcription in session state
            st.session_state["transcription_text"] = result["text"]
            # Reset annotation fields
            st.session_state["name"] = ""
            st.session_state["phone"] = ""
            st.session_state["address"] = ""
            st.session_state["notes"] = ""
        except Exception as e:
            st.error(f"Error during transcription: {e}")

    # Show annotation UI if transcription exists
    if "transcription_text" in st.session_state:
        transcription_text = st.session_state["transcription_text"]
        with st.form("annotation_form"):
            st.subheader("Annotate this entry")
            st.text_area(
                "Transcription",
                transcription_text,
                height=200,
                key="transcription",
                disabled=False,
            )
            name, phone = st.columns(2)
            with name:
                name_val = st.text_input(
                    "Name", value=st.session_state.get("name", ""), key="name"
                )
            with phone:
                phone_val = st.text_input(
                    "Phone", value=st.session_state.get("phone", ""), key="phone"
                )
            address = st.text_input(
                "Address", value=st.session_state.get("address", ""), key="address"
            )
            notes = st.text_area(
                "Notes", value=st.session_state.get("notes", ""), key="notes"
            )
            col_save, col_download = st.columns([1, 1])
            with col_save:
                save_clicked = st.form_submit_button("Save Annotation")
            with col_download:
                download_clicked = st.form_submit_button("Download as .txt")

        if save_clicked:
            entry = {
                "name": name_val,
                "phone": phone_val,
                "address": address,
                "notes": notes,
                "transcription": transcription_text,
            }
            save_entry(entry)
            st.success("Entry saved!")
        if download_clicked:
            entry = {
                "name": name_val,
                "phone": phone_val,
                "address": address,
                "notes": notes,
                "transcription": transcription_text,
            }
            txt = entry_to_txt(entry)
            st.download_button(
                label="Download Annotated Entry",
                data=txt,
                file_name=f"transcription_{name_val or 'entry'}.txt",
                mime="text/plain",
                key="download_current_entry",
            )

    # Show saved entries in the sidebar
    with st.sidebar:
        st.subheader("📋 Saved Entries (this session)")
        entries = st.session_state.get("entries", [])
        if not entries:
            st.info("No saved entries yet.")
        for i, entry in enumerate(entries):
            with st.expander(f"Entry {i+1}: {entry['name'] or 'No Name'}"):
                st.write(f"**Phone:** {entry['phone']}")
                st.write(f"**Address:** {entry['address']}")
                st.write(f"**Notes:** {entry['notes']}")
                st.write(f"**Transcription:**\n{entry['transcription']}\n")
                st.download_button(
                    label="Download This Entry as .txt",
                    data=entry_to_txt(entry),
                    file_name=f"transcription_{entry['name'] or 'entry'}.txt",
                    mime="text/plain",
                    key=f"download_{i}",
                )

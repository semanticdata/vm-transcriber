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
        st.text_area(
            "Transcription",
            transcription_text,
            height=200,
            key="transcription",
            disabled=True,
        )
        st.subheader("Annotate this entry:")
        name = st.text_input("Name", value=st.session_state.get("name", ""), key="name")
        phone = st.text_input(
            "Phone", value=st.session_state.get("phone", ""), key="phone"
        )
        address = st.text_input(
            "Address", value=st.session_state.get("address", ""), key="address"
        )
        notes = st.text_area(
            "Notes", value=st.session_state.get("notes", ""), key="notes"
        )

        # Save button
        if st.button("Save Annotation"):
            entry = {
                "name": name,
                "phone": phone,
                "address": address,
                "notes": notes,
                "transcription": transcription_text,
            }
            save_entry(entry)
            st.success("Entry saved!")

        # Download as .txt button
        if st.button("Download as .txt"):
            entry = {
                "name": name,
                "phone": phone,
                "address": address,
                "notes": notes,
                "transcription": transcription_text,
            }
            txt = entry_to_txt(entry)
            st.download_button(
                label="Download Annotated Entry",
                data=txt,
                file_name=f"transcription_{name or 'entry'}.txt",
                mime="text/plain",
            )

    # Show saved entries
    st.subheader("Saved Entries (this session):")
    entries = st.session_state.get("entries", [])
    for i, entry in enumerate(entries):
        with st.expander(f"Entry {i+1}: {entry['name'] or 'No Name'}"):
            st.write(f"**Phone:** {entry['phone']}")
            st.write(f"**Address:** {entry['address']}")
            st.write(f"**Notes:** {entry['notes']}")
            st.write(f"**Transcription:**\n{entry['transcription']}")
            st.download_button(
                label="Download This Entry as .txt",
                data=entry_to_txt(entry),
                file_name=f"transcription_{entry['name'] or 'entry'}.txt",
                mime="text/plain",
                key=f"download_{i}",
            )

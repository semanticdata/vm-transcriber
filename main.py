import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import sys
import contextlib

if sys.platform.startswith("win"):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import whisper
import tempfile

st.set_page_config(page_title="Speech-to-Text Demo")
st.title("🎤 Speech-to-Text Demo")
st.markdown(
    """
    <style>
        .stTooltipIcon div {
            min-width: 100%;
        }
        .stFileUploader > div {
            margin-block-start: 0.5rem;
            border: 1px solid var(--in-content-box-border-color);
            border-radius: 0.5rem;
        }
        .stForm button,
        .stFileUploader section button {
            width: 100%;
        }
        .stTooltipIcon div button {
            width: 100%;
            margin-block-start: 2.7rem;
        }
        .stHorizontalBlock {
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid var(--in-content-box-border-color);
        }
        .stForm {
            # border: none;
        }
        .stForm .stHorizontalBlock {
            border: none;
            padding: 0;
        }
        audio {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Constants ---
MODEL_OPTIONS = {
    "Tiny (~39MB)": "tiny",
    "Base (~74MB)": "base",
    "Small (~244MB)": "small",
    "Medium (~769MB)": "medium",
    "Large (~1.55GB)": "large",
}
LANGUAGES = {"English": "en", "Spanish": "es", "French": "fr"}


# --- Helper Functions ---
def get_session_list(key):
    """Get a list from session state, initializing if needed."""
    if key not in st.session_state:
        st.session_state[key] = []
    return st.session_state[key]


def get_session_value(key, default=None):
    """Get a value from session state, initializing if needed."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def set_session_value(key, value):
    """Set a value in session state."""
    st.session_state[key] = value


# --- UI Layout ---
col1, col2 = st.columns([2, 2])
with col1:
    uploaded_file = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "ogg", "flac", "webm"],
        accept_multiple_files=False,
    )
with col2:
    model_display = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=1)
    model_choice = MODEL_OPTIONS[model_display]
    language_display = st.selectbox("Language", list(LANGUAGES.keys()), index=0)
    language_hint = LANGUAGES[language_display]
    transcribe_clicked = st.button(
        "Transcribe",
        disabled=uploaded_file is None,
        help="Upload an audio file to enable transcription.",
    )


# --- Caching ---
@st.cache_resource(show_spinner=False)
def get_whisper_model(model_choice):
    """Load and cache the Whisper model by model_choice."""
    return whisper.load_model(model_choice)


@st.cache_data(show_spinner=False)
def save_entry(entry):
    """Save an annotated entry to session state."""
    entries = get_session_list("entries")
    entries.append(entry)
    set_session_value("entries", entries)
    return entries


def entry_to_txt(entry):
    """Format an entry as plain text for download."""
    return (
        f"Name: {entry['name']}\n"
        f"Phone: {entry['phone']}\n"
        f"Address: {entry['address']}\n"
        f"Notes: {entry['notes']}\n---\nTranscription:\n{entry['transcription']}\n"
    )


# --- Main Logic ---
if uploaded_file is not None:
    # Save to a temp file for whisper
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        with contextlib.suppress(Exception):
            tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Manual transcription trigger
    if transcribe_clicked:
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
            set_session_value("transcription_text", result["text"])
            set_session_value("name", "")
            set_session_value("phone", "")
            set_session_value("address", "")
            set_session_value("notes", "")
        except Exception as e:
            error_msg = str(e)
            if "not found; available models" in error_msg:
                # Extract available models from the error message
                available = error_msg.split("available models =", 1)[-1].strip()
                st.error(
                    f"The selected model '{model_choice}' was not found.\nAvailable models: {available}\nPlease select a valid model from the list."
                )
            else:
                st.error(f"An error occurred during transcription: {e}")

    # Show annotation UI if transcription exists
    transcription_text = st.session_state.get("transcription_text")
    if transcription_text:
        with st.form("annotation_form"):
            st.subheader("Annotate this entry")
            st.audio(uploaded_file, format=uploaded_file.type)
            st.text_area(
                "Transcription",
                transcription_text,
                height=200,
                key="transcription",
                disabled=False,
            )
            name_col, phone_col = st.columns(2)
            with name_col:
                name_val = st.text_input(
                    "Name", value=st.session_state.get("name", ""), key="name"
                )
            with phone_col:
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

        entry = {
            "name": name_val,
            "phone": phone_val,
            "address": address,
            "notes": notes,
            "transcription": transcription_text,
        }
        if save_clicked:
            save_entry(entry)
            st.success("Entry saved!")
        if download_clicked:
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
            with st.expander(f"Entry {i + 1}: {entry['name'] or 'No Name'}"):
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

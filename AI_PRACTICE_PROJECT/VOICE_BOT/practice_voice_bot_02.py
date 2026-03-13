import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import pyttsx3

# Settings
SAMPLE_RATE = 16000
DURATION = 4
AUDIO_FILE = "input.wav"

# Load Models
@st.cache_resource
def load_models():
    model = WhisperModel("base", compute_type="int8")
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    return model, engine

model, engine = load_models()

# Record Audio
def record_audio():
    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )
    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, recording)

# Speech to Text
def speech_to_text():
    segments, _ = model.transcribe(AUDIO_FILE)

    text = ""
    for segment in segments:
        text += segment.text

    return text.strip()

# Bot Conversation Logic
def get_response(text):

    text = text.lower()

    if "hi" in text or "hello" in text:
        return "Hello! It's nice to hear from you. How are you doing today?"

    elif "doing well" in text or "good" in text:
        return "That's great to hear! How can I assist you today?"

    elif "what can you do" in text:
        return "I can have simple conversations with you, answer basic questions, and demonstrate how a voice assistant works."

    elif "interesting" in text:
        return "I'm glad you think so! If you want, you can ask me something or just say goodbye."

    elif "bye" in text or "goodbye" in text:
        return "Goodbye! Have a great day ahead."

    else:
        return "Sorry, I didn't quite understand that. Could you please repeat?"

# Text to Speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("Friday - AI Voice Assistant")

st.write("Click the button below and start speaking with Friday.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Start Assistant "):

    greeting = "Hi, this is Friday! How can I help you today?"

    st.session_state.chat_history.append(("Friday", greeting))
    speak(greeting)

# Display chat history
for speaker, message in st.session_state.chat_history:
    if speaker == "Friday":
        st.markdown(f"** Friday:** {message}")
    else:
        st.markdown(f"** You:** {message}")

# Listen button
if st.button("Speak"):

    with st.spinner("Listening..."):
        record_audio()

    user_text = speech_to_text()

    st.session_state.chat_history.append(("You", user_text))

    response = get_response(user_text)

    st.session_state.chat_history.append(("Friday", response))

    speak(response)

    st.rerun()
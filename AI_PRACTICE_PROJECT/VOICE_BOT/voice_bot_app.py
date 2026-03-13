import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import pyttsx3
import ollama

from rag_engine import RAGEngine

SAMPLE_RATE = 16000
DURATION = 4
AUDIO_FILE = "input.wav"

# Load models =>
@st.cache_resource
def load_models():

    whisper = WhisperModel("base", compute_type="int8")

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    rag = RAGEngine()

    return whisper, engine, rag


whisper_model, tts_engine, rag_engine = load_models()

# Audio Recording =>
def record_audio():

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()

    write(AUDIO_FILE, SAMPLE_RATE, recording)

# Speech to Text =>
def speech_to_text():

    segments, _ = whisper_model.transcribe(AUDIO_FILE)

    text = ""

    for seg in segments:
        text += seg.text

    return text.strip()

# LLM Response with RAG =>
def generate_response(query):

    context = rag_engine.search(query)

    prompt = f"""
You are Friday, a helpful AI voice assistant.

Use the following context to answer the question.

Context:
{context}

User Question:
{query}

Answer clearly and concisely.
"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# TTS =>
def speak(text):

    tts_engine.say(text)
    tts_engine.runAndWait()

# Streamlit UI =>
st.title(" Friday - RAG Voice Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Start Assistant 🎤"):

    greeting = "Hi, this is Friday. How can I help you today?"

    st.session_state.history.append(("Friday", greeting))

    speak(greeting)

# Show conversation

for role, message in st.session_state.history:

    if role == "Friday":
        st.markdown(f"** Friday:** {message}")
    else:
        st.markdown(f"** You:** {message}")

# Speak button

if st.button("Speak"):

    with st.spinner("Listening..."):
        record_audio()

    user_text = speech_to_text()

    st.session_state.history.append(("You", user_text))

    response = generate_response(user_text)

    st.session_state.history.append(("Friday", response))

    speak(response)

    st.rerun()
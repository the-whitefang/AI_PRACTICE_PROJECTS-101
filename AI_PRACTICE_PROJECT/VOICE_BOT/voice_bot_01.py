import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import pyttsx3
import os


# Settings
SAMPLE_RATE = 16000
DURATION = 4
AUDIO_FILE = "input.wav"

# Load Models
@st.cache_resource
def load_models():
    whisper_model = WhisperModel("base", compute_type="int8")
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    return whisper_model, engine


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

    segments, info = model.transcribe(AUDIO_FILE)

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()


# Bot Logic
def get_response(text):

    text = text.lower()

    if "hi" in text or "hello" in text:
        return "How are you? Are you doing fine?"

    elif "how are you" in text:
        return "I am doing great. Thanks for asking."

    elif "bye" in text:
        return "Goodbye. Have a great day."

    else:
        return "Sorry, I did not understand that."


# Text to Speech
def speak(text):

    engine.say(text)
    engine.runAndWait()


# Streamlit UI
st.title("🎤 Simple Voice Bot")

st.write("Click the button and speak into the microphone.")

if st.button("Start Listening"):

    with st.spinner("Listening... Speak now"):

        record_audio()

        user_text = speech_to_text()

    st.success("Audio Processed")

    st.write("### You said:")
    st.write(user_text)

    response = get_response(user_text)

    st.write("### Bot Response:")
    st.write(response)

    speak(response)
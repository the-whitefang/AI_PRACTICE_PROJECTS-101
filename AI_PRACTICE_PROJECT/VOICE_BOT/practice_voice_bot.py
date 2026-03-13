import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import pyttsx3
import time

# -----------------------------
# Settings
# -----------------------------

SAMPLE_RATE = 16000
DURATION = 4
AUDIO_FILE = "input.wav"

# -----------------------------
# Load Models
# -----------------------------

print("Loading Whisper model...")
model = WhisperModel("base", compute_type="int8")

print("Initializing TTS engine...")
engine = pyttsx3.init()

# Adjust voice speed if needed
engine.setProperty("rate", 170)

print("Voice bot ready!")

# -----------------------------
# Record Audio
# -----------------------------

def record_audio():
    print("\nSpeak now...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1)
    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, recording)
    print("Recording complete.")


# -----------------------------
# Speech To Text
# -----------------------------

def speech_to_text():
    segments, info = model.transcribe(AUDIO_FILE)

    text = ""
    for segment in segments:
        text += segment.text

    return text.strip()


# -----------------------------
# Bot Logic
# -----------------------------

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


# -----------------------------
# Text To Speech
# -----------------------------

def speak(text):

    print("Bot:", text)

    engine.say(text)
    engine.runAndWait()


# -----------------------------
# Main Loop
# -----------------------------

while True:

    record_audio()

    user_text = speech_to_text()

    print("You said:", user_text)

    response = get_response(user_text)

    speak(response)

    if "bye" in user_text.lower():
        break

    time.sleep(1)
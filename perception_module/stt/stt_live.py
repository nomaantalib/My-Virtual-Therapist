
# stt/stt_live.py
"""
Speech-to-text live processing module.
Handles audio recording, noise reduction, saving, and transcription.
"""
import sounddevice as sd
import requests
import time
import wave
import numpy as np
import noisereduce as nr  # type: ignore
import asyncio
from config import API_KEY   # import from root config

stop_stream = False

def record_audio(duration=10) -> np.ndarray:
    fs = 16000
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return recording

def reduce_noise(audio: np.ndarray, fs=16000) -> np.ndarray:
    audio_float = audio.astype(np.float32)
    reduced_noise = nr.reduce_noise(y=audio_float.flatten(), sr=fs)
    return reduced_noise.astype(np.int16)

def save_wav(data: np.ndarray, filename: str):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())

def transcribe_audio(filename: str) -> str:
    """
    Upload audio file to AssemblyAI and get transcript text.
    Raises Exception on transcription error.
    """
    headers = {"authorization": API_KEY}
    with open(filename, 'rb') as f:
        response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    upload_url = response.json()["upload_url"]

    transcript_request = {
        "audio_url": upload_url
    }
    response = requests.post("https://api.assemblyai.com/v2/transcript", json=transcript_request, headers=headers)
    transcript_id = response.json()["id"]

    while True:
        response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        data = response.json()
        if data["status"] == "completed":
            return data["text"]
        elif data["status"] == "error":
            raise Exception(data["error"])
        time.sleep(1)

async def start_stt(callback, duration=5):
    """
    Start live STT with noise reduction and callback on transcript.
    Runs until global stop_stream is set to True.
    """
    global stop_stream
    while not stop_stream:
        audio = record_audio(duration)
        reduced_audio = reduce_noise(audio)
        save_wav(reduced_audio, 'temp.wav')
        text = transcribe_audio('temp.wav')
        callback(text, reduced_audio)
        await asyncio.sleep(0.1)

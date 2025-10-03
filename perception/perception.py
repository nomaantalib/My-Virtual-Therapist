"""
Perception module: Combines STT, Tone, NLU, and Facial Emotion analysis.
"""
import tempfile
import os
import numpy as np
from .stt.stt_live import save_wav, transcribe_audio
from .tone.tone_sentiment_live import analyze_tone
from .nlu.nlu_live import nlu_process
from .facial.facial_emotion import analyze_facial_emotion

class Perception:
    def __init__(self):
        pass

    def process(self, audio_data: np.ndarray, image=None) -> dict:
        """
        Process audio and optional image through perception modules.
        Returns combined result dict.
        """
        # Save audio to temp file for STT
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_filename = f.name
        save_wav(audio_data, temp_filename)

        try:
            # STT
            transcript = transcribe_audio(temp_filename)

            # Tone
            tone = analyze_tone(transcript, audio_data)

            # NLU
            nlu_result = nlu_process(transcript, tone)

            # Facial Emotion
            facial_emotion = analyze_facial_emotion(image)

            # Combine
            result = {
                "transcript": transcript,
                "tone": tone,
                "nlu": nlu_result,
                "facial_emotion": facial_emotion
            }

            return result
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

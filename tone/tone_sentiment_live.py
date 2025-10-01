# tone/tone_sentiment_live.py
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import librosa  # type: ignore
import numpy as np

sia = SentimentIntensityAnalyzer()

emotion_lexicon = {
    "happy": ["happy", "joy", "excited", "delighted", "pleased", "cheerful"],
    "sad": ["sad", "unhappy", "depressed", "sorrow", "grief", "miserable"],
    "angry": ["angry", "mad", "furious", "irritated", "annoyed", "rage"],
    "fear": ["fear", "scared", "afraid", "terrified", "anxious", "worried"],
    "surprise": ["surprise", "shocked", "amazed", "astonished", "startled"],
    "disgust": ["disgust", "repulsed", "gross", "nauseated", "disgusted"],
    "confused": ["confused", "bewildered", "puzzled", "lost", "uncertain"],
    "hesitant": ["hesitant", "unsure", "doubtful", "reluctant", "wavering"]
}

question_words = ["who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might"]

def detect_emotions(text: str) -> list:
    tokens = nltk.word_tokenize(text.lower())
    detected = [emotion for emotion, keywords in emotion_lexicon.items() if any(word in tokens for word in keywords)]
    return detected if detected else ["neutral"]

def is_questioning(text: str) -> bool:
    if '?' in text:
        return True
    tokens = nltk.word_tokenize(text.lower())
    return any(word in tokens for word in question_words)

def analyze_pitch(audio, sr=16000):
    if audio is None:
        return {"mean_pitch": 0, "std_pitch": 0, "pitch_range": 0}
    audio_float = audio.astype(np.float32) / 32768.0
    f0, voiced_flag, _ = librosa.pyin(
        audio_float,
        fmin=float(librosa.note_to_hz('C2')),
        fmax=float(librosa.note_to_hz('C7')),
        sr=sr
    )
    f0 = f0[voiced_flag]
    if len(f0) == 0:
        return {"mean_pitch": 0, "std_pitch": 0, "pitch_range": 0}
    return {
        "mean_pitch": float(np.mean(f0)),
        "std_pitch": float(np.std(f0)),
        "pitch_range": float(np.max(f0) - np.min(f0))
    }

def analyze_tone(text: str, audio=None) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # type: ignore
    subjectivity = blob.sentiment.subjectivity  # type: ignore
    compound = sia.polarity_scores(text)['compound']
    emotions = detect_emotions(text)
    overall_mood = "positive" if polarity > 0.1 or compound > 0.1 else "negative" if polarity < -0.1 or compound < -0.1 else "neutral"
    annotations = [ann for ann in ["questioning", "confused", "hesitant"] if (ann == "questioning" and is_questioning(text)) or (ann in emotions)]
    pitch_stats = analyze_pitch(audio)
    return {
        "sentiment": {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "compound_score": compound
        },
        "emotions": emotions,
        "overall_mood": overall_mood,
        "annotations": annotations,
        "pitch_stats": pitch_stats
    }

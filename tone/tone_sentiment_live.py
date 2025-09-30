# tone/tone_sentiment_live.py
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import librosa  # type: ignore
import numpy as np

# Initialize VADER for sentiment analysis
sia = SentimentIntensityAnalyzer()

# Expanded emotion lexicon for keyword-based detection
emotion_lexicon = {
    "happy": ["happy", "joy", "excited", "delighted", "pleased", "cheerful", "glad", "thrilled", "ecstatic"],
    "sad": ["sad", "unhappy", "depressed", "sorrow", "grief", "miserable", "down", "blue", "melancholy"],
    "angry": ["angry", "mad", "furious", "irritated", "annoyed", "rage", "frustrated", "outraged"],
    "fear": ["fear", "scared", "afraid", "terrified", "anxious", "worried", "nervous", "panicked"],
    "surprise": ["surprise", "shocked", "amazed", "astonished", "startled", "bewildered"],
    "disgust": ["disgust", "repulsed", "gross", "nauseated", "disgusted", "revulsion"],
    "confused": ["confused", "bewildered", "puzzled", "lost", "uncertain", "baffled", "perplexed"],
    "hesitant": ["hesitant", "unsure", "doubtful", "reluctant", "wavering", "indecisive"]
}

# Words that indicate questioning
question_words = ["who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "will", "would", "should", "may", "might"]

def detect_emotions(text: str) -> list:
    """
    Detect emotions based on keyword matching.
    """
    tokens = nltk.word_tokenize(text.lower())
    detected = set()
    for emotion, keywords in emotion_lexicon.items():
        if any(word in tokens for word in keywords):
            detected.add(emotion)
    return list(detected) if detected else ["neutral"]

def is_questioning(text: str) -> bool:
    """
    Detect if the text is a question.
    """
    if '?' in text:
        return True
    tokens = nltk.word_tokenize(text.lower())
    return any(word in tokens for word in question_words)

def analyze_pitch(audio, sr=16000):
    """
    Analyze pitch from audio numpy array to extract mean, std, and range.
    """
    if audio is None:
        return {"mean_pitch": 0, "std_pitch": 0, "pitch_range": 0}
    # Normalize audio to float32 in range -1 to 1
    audio_float = audio.astype(np.float32) / 32768.0
    # Extract fundamental frequency using probabilistic YIN
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_float,
        fmin=float(librosa.note_to_hz('C2')),
        fmax=float(librosa.note_to_hz('C7')),
        sr=sr
    )
    # Filter to voiced frames
    f0 = f0[voiced_flag]
    if len(f0) == 0:
        return {"mean_pitch": 0, "std_pitch": 0, "pitch_range": 0}
    mean_pitch = float(np.mean(f0))
    std_pitch = float(np.std(f0))
    pitch_range = float(np.max(f0) - np.min(f0))
    return {"mean_pitch": mean_pitch, "std_pitch": std_pitch, "pitch_range": pitch_range}

def analyze_tone(text: str, audio=None) -> dict:
    """
    Enhanced tone & sentiment analysis for therapeutic context.
    Combines TextBlob, VADER, keyword-based emotion detection, and pitch analysis.
    """
    # Sentiment analysis using TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment
    polarity = sentiment.polarity  # type: ignore
    subjectivity = sentiment.subjectivity  # type: ignore

    # VADER sentiment scores for more nuanced analysis
    vader_scores = sia.polarity_scores(text)
    compound = vader_scores['compound']

    # Detect emotions based on keywords
    emotions = detect_emotions(text)

    # Determine overall mood based on polarity and compound score
    if polarity > 0.1 or compound > 0.1:
        overall_mood = "positive"
    elif polarity < -0.1 or compound < -0.1:
        overall_mood = "negative"
    else:
        overall_mood = "neutral"

    # Detect annotations like questioning, confused, hesitant
    annotations = []
    if is_questioning(text):
        annotations.append("questioning")
    if "confused" in emotions:
        annotations.append("confused")
    if "hesitant" in emotions:
        annotations.append("hesitant")

    # Pitch analysis from audio
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

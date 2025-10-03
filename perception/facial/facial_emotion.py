from deepface import DeepFace
import numpy as np
from PIL import Image

def analyze_facial_emotion(image=None):
    """
    Analyze facial emotion from provided image.
    If no image, return error.
    """
    if image is None:
        return {"error": "No image provided"}

    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            frame = np.array(image)
        else:
            frame = image

        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        if isinstance(analysis, list) and len(analysis) > 0:
            result = analysis[0]
        elif isinstance(analysis, dict):
            result = analysis
        else:
            return {"error": "Analysis failed"}

        if 'emotion' in result:
            return result['emotion']
        else:
            return {"error": "No emotion data"}
    except Exception as e:
        return {"error": str(e)}

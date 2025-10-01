import cv2
from deepface import DeepFace

import time
import numpy as np
import cv2
from deepface import DeepFace

def analyze_facial_emotion():
    cap = cv2.VideoCapture(0)  # open webcam
    if not cap.isOpened():
        return {"error": "Camera not accessible"}

    emotions_accum = {}
    frame_count = 0
    max_frames = 3  # small number for averaging

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            first_result = analysis[0] if isinstance(analysis, list) and len(analysis) > 0 else {}
            if 'emotion' in first_result:
                emotions = first_result['emotion']
                for k, v in emotions.items():
                    emotions_accum[k] = emotions_accum.get(k, 0) + float(v)
                frame_count += 1
            time.sleep(0.1)  # small delay between frames
        cap.release()

        if frame_count == 0:
            return {"error": "No frames analyzed"}

        # Average the emotions
        averaged_emotions = {k: v / frame_count for k, v in emotions_accum.items()}
        return averaged_emotions
    except Exception as e:
        cap.release()
        return {"error": str(e)}

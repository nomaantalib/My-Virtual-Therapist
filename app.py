import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

import tempfile
import os
import base64
import io
from PIL import Image
import numpy as np
from deepface import DeepFace
from perception_module.stt.stt_live import save_wav, transcribe_audio
from perception_module.tone.tone_sentiment_live import analyze_tone
from perception_module.nlu.nlu_live import nlu_process
from perception_module.facial.facial_emotion import analyze_facial_emotion
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory

# Initialize Flask app
app = Flask(__name__)
socketio = SocketIO(app)

# Initialize memory modules
working_memory = WorkingMemory()
long_term_memory = LongTermMemory()
logs = []

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

def process_audio(audio_file):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_filename = f.name
    audio_file.save(temp_filename)
    try:
        transcript = transcribe_audio(temp_filename)
        logs.append(f"Transcribed: {transcript}")
        tone = analyze_tone(transcript)
        logs.append(f"Tone analyzed: {tone}")
        facial_emotion = {}
        logs.append(f"Facial emotion: {facial_emotion}")
        result = nlu_process(transcript, tone)
        logs.append(f"NLU result: {result}")
        result['facial_emotion'] = facial_emotion
        working_memory.add_interaction(transcript, str(result), [])
        logs.append("Added to working memory")
        if len(working_memory.get_context()) >= working_memory.max_length:
            long_term_memory.store_interaction("user", transcript, str(result), [])
            logs.append("Stored in long term memory")
            working_memory.clear()
            logs.append("Cleared working memory")
        return result
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    try:
        result = process_audio(audio_file)
        return jsonify(result)
    except Exception as e:
        logs.append(f"Error: {str(e)}")
        app.logger.error(f"Error in analyze: {str(e)}")
        return jsonify({"error": "An error occurred during analysis"}), 500

@app.route('/get_working_history')
def get_working_history():
    context = working_memory.get_context()
    return jsonify(context)

@app.route('/get_long_term_history')
def get_long_term_history():
    # history = long_term_memory.retrieve_user_history("user", limit=20)
    return jsonify([])

@app.route('/get_logs')
def get_logs():
    return jsonify(logs[-50:])  # Last 50 logs

@socketio.on('audio_data')
def handle_audio_data(data):
    app.logger.info("Received audio data event")
    # data is dict with 'audio' and 'image' base64
    audio_b64 = data.get('audio')
    image_b64 = data.get('image')
    if not audio_b64:
        emit('analysis_error', {'error': 'No audio data'})
        return
    audio_data = base64.b64decode(audio_b64)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_filename = f.name
        f.write(audio_data)
    try:
        app.logger.info(f"Saved audio to temp file {temp_filename}")
        transcript = transcribe_audio(temp_filename)
        logs.append(f"Transcribed: {transcript}")
        app.logger.info(f"Transcribed text: {transcript}")
        tone = analyze_tone(transcript)
        logs.append(f"Tone analyzed: {tone}")
        app.logger.info(f"Tone analysis: {tone}")
        facial_emotion = {}
        if image_b64:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            facial_emotion = analyze_facial_emotion(image)
        logs.append(f"Facial emotion: {facial_emotion}")
        app.logger.info(f"Facial emotion: {facial_emotion}")
        result = nlu_process(transcript, tone)
        logs.append(f"NLU result: {result}")
        app.logger.info(f"NLU result: {result}")
        result['facial_emotion'] = facial_emotion
        working_memory.add_interaction(transcript, str(result), [])
        logs.append("Added to working memory")
        app.logger.info("Added interaction to working memory")
        # Add transcript to result for client display
        result['transcript'] = transcript
        # Add tone and nlu details separately for better UI display
        result['tone'] = tone
        emit('analysis_result', result)
    except Exception as e:
        app.logger.error(f"Error processing audio data: {str(e)}")
        logs.append(f"Error: {str(e)}")
        emit('analysis_error', {'error': str(e)})
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

@app.route('/analyze_facial', methods=['POST'])
def analyze_facial():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        facial_emotion = analyze_facial_emotion(image)
        return jsonify({'facial_emotion': facial_emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)

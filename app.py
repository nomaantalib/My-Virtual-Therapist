from flask import Flask, render_template, jsonify, request
import tempfile
import os
from stt.stt_live import save_wav, transcribe_audio
from tone.tone_sentiment_live import analyze_tone
from nlu.nlu_live import nlu_process
from facial.facial_emotion import analyze_facial_emotion

# Initialize Flask app
app = Flask(__name__)

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
        tone = analyze_tone(transcript)
        facial_emotion = analyze_facial_emotion()
        result = nlu_process(transcript, tone)
        result['facial_emotion'] = facial_emotion
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
        app.logger.error(f"Error in analyze: {str(e)}")
        return jsonify({"error": "An error occurred during analysis"}), 500

if __name__ == '__main__':
    app.run(debug=True)

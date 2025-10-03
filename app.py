from flask import Flask, render_template, jsonify, request
import tempfile
import os
from perception.stt.stt_live import transcribe_audio
from perception.tone.tone_sentiment_live import analyze_tone
from perception.nlu.nlu_live import nlu_process
from perception.facial.facial_emotion import analyze_facial_emotion
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory

app = Flask(__name__)

# Initialize memory modules
wm = WorkingMemory()
ltm = LongTermMemory()

# In-memory logs for display
wm_logs = []
ltm_logs = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Received /analyze POST request")
    try:
        text = request.form.get('text')
        if text:
            print(f"Using provided text: {text}")
            transcript = text
        else:
            if 'audio' not in request.files:
                print("No audio file provided")
                return jsonify({"error": "No audio file provided"}), 400
            audio_file = request.files['audio']
            if audio_file.filename == '':
                print("No audio file selected")
                return jsonify({"error": "No audio file selected"}), 400

            print("Saving audio to temp file")
            # Save audio to temp wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                audio_filename = f.name
            audio_file.save(audio_filename)

            print("Transcribing audio")
            # Transcribe audio
            transcript = transcribe_audio(audio_filename)
            print(f"Transcript: {transcript}")
            os.unlink(audio_filename)

        print("Analyzing tone")
        # Analyze tone
        tone = analyze_tone(transcript)
        print(f"Tone: {tone}")

        print("NLU processing")
        # NLU processing
        perception_result = nlu_process(transcript, tone)
        print(f"Perception result: {perception_result}")

        print("Analyzing facial emotion")
        # Analyze facial emotion
        facial_emotion = {}
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_file.read()))
                facial_emotion = analyze_facial_emotion(image)
        print(f"Facial emotion: {facial_emotion}")

        # Convert any numpy float32 in facial_emotion to float for JSON serialization
        def convert_floats(obj):
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(i) for i in obj]
            elif 'numpy' in str(type(obj)):
                return float(obj)
            else:
                return obj

        perception_result['facial_emotion'] = convert_floats(facial_emotion)

        print("Storing in working memory")
        # Store in working memory
        wm.add_interaction(
            user_input=transcript,
            system_output=str(perception_result),
            emotion_tags=convert_floats(perception_result.get("facial_emotion"))
        )

        print("Storing in long term memory")
        # Store in long term memory
        user_id = "default_user"
        ltm.store_interaction(
            user_id=user_id,
            user_input=transcript,
            system_output=str(perception_result),
            emotion_tags=convert_floats(perception_result.get("facial_emotion"))
        )

        print("Returning response")
        return jsonify({
            "perception": perception_result,
            "working_memory": wm.get_context(),
            "long_term_memory": ltm.retrieve_user_history(user_id, limit=10)
        })
    except Exception as e:
        print(f"Error in analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

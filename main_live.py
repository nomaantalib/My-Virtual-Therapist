# main_live.py
"""
Main script for live speech-to-text processing with tone analysis and NLU.
Runs STT in a loop, analyzes tone, and processes with NLU.
"""
import asyncio
import threading
import sys

# Import modules from subfolders
from perception_module.stt.stt_live import start_stt, stop_stream
from perception_module.tone.tone_sentiment_live import analyze_tone
from perception_module.nlu.nlu_live import nlu_process
from perception_module.facial.facial_emotion import analyze_facial_emotion
from memory.working_memory import WorkingMemory
from memory.long_term_memory import LongTermMemory

# Initialize memory modules
working_memory = WorkingMemory()
long_term_memory = LongTermMemory()
interaction_counter = 0

def handle_text(text: str, audio):
    tone = analyze_tone(text, audio)
    facial_emotion = analyze_facial_emotion()
    result = nlu_process(text, tone)
    working_memory.add_interaction(text, str(result), [])
    global interaction_counter
    interaction_counter += 1
    if interaction_counter >= 5:
        session_id = f"live_session_{interaction_counter // 5}"
        for interaction in working_memory.get_context():
            long_term_memory.store_interaction("user", interaction["user_input"], interaction["system_output"], interaction["emotion_tags"])
        working_memory.clear()
        interaction_counter = 0
    print("\n🗣️ Transcript:", text)
    print("🤖 AGI Response:", result)
    print("😊 Facial Emotion:", facial_emotion)

def listen_for_quit():
    print("\nPress 'q' + Enter anytime to quit...\n")
    while True:
        key = sys.stdin.readline().strip().lower()
        if key == "q":
            global stop_stream
            stop_stream = True
            print("🛑 Stopping transcription...")
            break

if __name__ == "__main__":
    # Start background thread for quit listener
    threading.Thread(target=listen_for_quit, daemon=True).start()
    # Run the STT loop asynchronously
    asyncio.run(start_stt(handle_text))

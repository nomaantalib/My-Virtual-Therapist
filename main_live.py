# main_live.py
"""
Main script for live speech-to-text processing with tone analysis and NLU.
Runs STT in a loop, analyzes tone, and processes with NLU.
"""
import asyncio
import threading
import sys

from integration import Integration
from perception.stt.stt_live import start_stt

# Initialize integrated system
integration = Integration()

def handle_text(text: str, audio):
    result = integration.process_input(audio)
    print("\n🗣️ Transcript:", result.get("transcript", ""))
    print("🤖 AGI Response:", result.get("nlu", {}))
    print("😊 Facial Emotion:", result.get("facial_emotion", {}))

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

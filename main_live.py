# main_live.py
"""
Main script for live speech-to-text processing with tone analysis and NLU.
Runs STT in a loop, analyzes tone, and processes with NLU.
"""
import asyncio
import threading
import sys

# Import modules from subfolders
from stt.stt_live import start_stt, stop_stream
from tone.tone_sentiment_live import analyze_tone
from nlu.nlu_live import nlu_process

def handle_text(text: str, audio):
    tone = analyze_tone(text, audio)
    result = nlu_process(text, tone)
    print("\n🗣️ Transcript:", text)
    print("🤖 AGI Response:", result)

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

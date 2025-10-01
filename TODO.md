# Refactoring Plan for AGI Therapist Perception Module

## Tasks
- [x] Refactor tone/tone_sentiment_live.py: Reduce emotion_lexicon to essential emotions, simplify detect_emotions using list comprehension, optimize analyze_tone logic.
- [x] Refactor nlu/nlu_live.py: Use list comprehensions for get_entities and get_roles to reduce LOC.
- [x] Refactor stt/stt_live.py: Minor simplifications, e.g., combine record and reduce noise if possible.
- [x] Refactor app.py: Extract the analysis logic from the analyze route into a helper function.
- [x] Refactor main_live.py: Simplify the quit listener or integrate better.

## Progress
- [x] Created TODO.md
- [x] Refactored tone/tone_sentiment_live.py

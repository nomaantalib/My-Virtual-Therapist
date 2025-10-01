from collections import deque

class WorkingMemory:
    def __init__(self, max_length=10):
        self.max_length = max_length
        self.context = deque(maxlen=max_length)
        self.current_state = {}

    def add_interaction(self, user_input, system_output, emotion_tags=None):
        self.context.append({
            "user_input": user_input,
            "system_output": system_output,
            "emotion_tags": emotion_tags
        })

    def get_context(self):
        return list(self.context)

    def set_state(self, key, value):
        self.current_state[key] = value

    def get_state(self, key):
        return self.current_state.get(key, None)

    def clear(self):
        self.context.clear()
        self.current_state = {}

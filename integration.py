"""
Integration module: Integrates perception and memory modules.
"""
import json
from perception.perception import Perception
from memory.memory import Memory

class Integration:
    def __init__(self):
        self.perception = Perception()
        self.memory = Memory()

    def process_input(self, audio_data, image=None):
        """
        Process input through perception and update memory.
        Returns combined perception result.
        """
        perception_result = self.perception.process(audio_data, image)
        transcript = perception_result.get("transcript", "")
        nlu_result = perception_result.get("nlu", {})
        tone = perception_result.get("tone", {})
        facial_emotion = perception_result.get("facial_emotion", {})

        # Combine emotion tags from NLU, tone, and facial
        emotion_tags = nlu_result.get("emotions", []) + [tone.get("sentiment", "")] + [facial_emotion.get("emotion", "")]

        # Store full perception result as system_output for richer context
        system_output = json.dumps(perception_result)

        # Add interaction to memory
        self.memory.add_interaction(transcript, system_output, emotion_tags)

        return perception_result

    def get_working_memory_context(self):
        return self.memory.get_context()

    def get_long_term_memory_history(self, limit=50):
        return self.memory.get_long_term_history(limit)

    def search_memory(self, query, n=5):
        return self.memory.search_similar(query, n)

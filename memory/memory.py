"""
Memory module: Combines Working Memory and Long Term Memory.
"""
from .working_memory import WorkingMemory
from .long_term_memory import LongTermMemory

class Memory:
    def __init__(self, user_id="user"):
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory()
        self.user_id = user_id
        self.interaction_counter = 0

    def add_interaction(self, user_input: str, system_output: str, emotion_tags=None):
        """
        Add an interaction to working memory.
        If working memory is full, transfer to long term memory.
        """
        self.working_memory.add_interaction(user_input, system_output, emotion_tags or [])
        self.interaction_counter += 1
        if self.interaction_counter >= self.working_memory.max_length:
            # Transfer to long term
            for interaction in self.working_memory.get_context():
                self.long_term_memory.store_interaction(
                    self.user_id,
                    interaction["user_input"],
                    interaction["system_output"],
                    interaction["emotion_tags"]
                )
            self.working_memory.clear()
            self.interaction_counter = 0

    def get_context(self):
        """
        Get current context from working memory.
        """
        return self.working_memory.get_context()

    def get_long_term_history(self, limit=50):
        """
        Get long term history for the user.
        """
        return self.long_term_memory.retrieve_user_history(self.user_id, limit=limit)

    def search_similar(self, query, n=5):
        """
        Search for similar interactions in long term memory.
        """
        return self.long_term_memory.search_similar_interactions(query, self.user_id, n)

    def set_state(self, key, value):
        """
        Set a state in working memory.
        """
        self.working_memory.set_state(key, value)

    def get_state(self, key):
        """
        Get a state from working memory.
        """
        return self.working_memory.get_state(key)

    def clear_working(self):
        """
        Clear working memory.
        """
        self.working_memory.clear()
        self.interaction_counter = 0

    def get_combined_context(self, query=None, n=5):
        """
        Get combined context: working memory + relevant long-term memory.
        If query is provided, search LTM for similar interactions.
        """
        context = self.get_context()  # Working memory
        if query:
            ltm_results = self.search_similar(query, n)
            # Extract documents from ltm_results
            ltm_context = []
            if ltm_results and 'documents' in ltm_results and ltm_results['documents']:
                for doc in ltm_results['documents'][0]:
                    ltm_context.append(doc)
            context.extend(ltm_context)
        else:
            # Get recent LTM history
            ltm_history = self.get_long_term_history(limit=n)
            context.extend([f"User: {h['user_input']} Therapist: {h['system_output']}" for h in ltm_history])
        return context

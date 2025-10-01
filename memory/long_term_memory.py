import chromadb
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    def __init__(self, collection_name="therapy_memory"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = None

    def _get_embedder(self):
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Error loading embedder: {e}")
                # Fallback to simple hash or something, but for now, raise
                raise e
        return self.embedder

    def store_interaction(self, user_id, user_input, system_output, emotion_tags=None):
        # Create a unique ID for the interaction
        interaction_id = f"{user_id}_{datetime.now().isoformat()}"
        # Combine user_input and system_output for embedding
        text = f"User: {user_input} Therapist: {system_output}"
        embedder = self._get_embedder()
        embedding = embedder.encode([text])[0].tolist()
        metadata = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "system_output": system_output,
            "emotion_tags": json.dumps(emotion_tags) if emotion_tags else None
        }
        self.collection.add(documents=[text], embeddings=[embedding], metadatas=[metadata], ids=[interaction_id])

    def retrieve_user_history(self, user_id, limit=50):
        # Retrieve interactions for the user
        results = self.collection.get(where={"user_id": user_id}, limit=limit)
        history = []
        for i in range(len(results['ids'])):
            metadata = results['metadatas'][i]
            history.append({
                "timestamp": metadata["timestamp"],
                "user_input": metadata["user_input"],
                "system_output": metadata["system_output"],
                "emotion_tags": json.loads(metadata["emotion_tags"]) if metadata["emotion_tags"] else None
            })
        # Sort by timestamp descending
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history

    def search_similar_interactions(self, query, user_id=None, n=5):
        # Search for similar interactions
        embedder = self._get_embedder()
        embedding = embedder.encode([query])[0].tolist()
        where_clause = {"user_id": user_id} if user_id else None
        results = self.collection.query(query_embeddings=[embedding], n_results=n, where=where_clause)
        return results

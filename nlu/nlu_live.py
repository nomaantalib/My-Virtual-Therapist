# nlu/nlu_live.py
"""
Natural Language Understanding module.
Handles entity extraction and semantic role labeling.
"""
import nltk

def setup_nltk():
    """
    Download required NLTK data quietly.
    Call this function before using NLTK functions.
    """
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    nltk.download("maxent_ne_chunker_tab", quiet=True)
    nltk.download("words", quiet=True)

# Initialize NLTK data
setup_nltk()

def get_entities(text: str) -> list:
    try:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tags)
        return [{"entity": " ".join([word for word, tag in subtree.leaves()]), "type": subtree.label()} for subtree in tree if isinstance(subtree, nltk.Tree)]
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []

def get_roles(text: str) -> list:
    try:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        return [{"word": word, "role": "entity" if tag.startswith("NN") else "action"} for word, tag in tags if tag.startswith("NN") or tag.startswith("VB")]
    except Exception as e:
        print(f"Error extracting roles: {e}")
        return []

def nlu_process(text: str, tone_obj: dict) -> dict:
    """
    Process text with NLU: extract entities and roles.
    Combine with tone analysis results.
    """
    return {
        "transcript": text,
        "sentiment": tone_obj["sentiment"],
        "emotions": tone_obj["emotions"],
        "overall_mood": tone_obj.get("overall_mood", "neutral"),
        "annotations": tone_obj.get("annotations", []),
        "pitch_stats": tone_obj.get("pitch_stats", {}),
        "entities": get_entities(text),
        "semantic_roles": get_roles(text)
    }

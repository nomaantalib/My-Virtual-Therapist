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
    """
    Extract named entities from text using NLTK.
    Tokenizes text, tags parts of speech, and chunks named entities.
    Returns list of dicts with 'entity' and 'type' keys.
    Handles errors gracefully by returning empty list.
    """
    try:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        tree = nltk.ne_chunk(tags)
        entities = []
        for subtree in tree:
            if isinstance(subtree, nltk.Tree):
                entity = " ".join([word for word, tag in subtree.leaves()])
                label = subtree.label()
                entities.append({"entity": entity, "type": label})
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []

def get_roles(text: str) -> list:
    """
    Extract semantic roles based on POS tags.
    Classifies nouns as 'entity' and verbs as 'action'.
    Returns list of dicts with 'word' and 'role'.
    Handles errors by returning empty list.
    """
    try:
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        roles = []
        for word, tag in tags:
            if tag.startswith("NN"):
                roles.append({"word": word, "role": "entity"})
            elif tag.startswith("VB"):
                roles.append({"word": word, "role": "action"})
        return roles
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

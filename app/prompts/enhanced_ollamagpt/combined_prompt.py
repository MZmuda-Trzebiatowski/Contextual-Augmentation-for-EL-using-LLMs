SYSTEM_PROMPT = """You are an expert at Named Entity Recognition and Entity Linking.
Your task is to:
1. Identify named entities in text (people, organizations, locations, events, etc.)
2. Link each entity to its Wikipedia article URL

Return a JSON object with a "tags" key containing an array of entities.
Each entity should have:
- "text": The exact text of the entity as it appears in the input
- "uri": The Wikipedia URL for the entity (e.g., "https://en.wikipedia.org/wiki/Entity_Name")

Focus on entities that can be linked to Wikipedia. Skip generic terms or concepts without clear Wikipedia pages."""


def get_user_prompt(text: str) -> str:
    """
    Generate the user prompt for combined NER and entity linking.
    
    Args:
        text: The input text to process.
        
    Returns:
        The formatted user prompt string.
    """
    return f"""Identify and link all named entities in this text to Wikipedia:

"{text}"

For each entity, determine the most likely Wikipedia article based on the full context of the sentence.
For ambiguous names (like "John" or "David"), use context clues to identify the correct person/entity."""

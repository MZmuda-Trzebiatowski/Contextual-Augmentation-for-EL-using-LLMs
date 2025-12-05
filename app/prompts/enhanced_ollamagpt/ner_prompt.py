SYSTEM_PROMPT = """You are an expert Named Entity Recognition system.
Tag named entities using [START_ENT] and [END_ENT] markers.
Focus on: people, organizations, locations, events, works of art, products.
Do NOT tag common nouns, adjectives, or generic terms.
Return ONLY the tagged text, no explanations."""


def get_user_prompt(text: str) -> str:
    """
    Generate the user prompt for NER.
    
    Args:
        text: The input text to perform NER on.
        
    Returns:
        The formatted user prompt string.
    """
    return f"""Tag all named entities in this text:

{text}

Example:
Input: "Einstein worked at Princeton University."
Output: "[START_ENT]Einstein[END_ENT] worked at [START_ENT]Princeton University[END_ENT]."

Now tag the entities:"""

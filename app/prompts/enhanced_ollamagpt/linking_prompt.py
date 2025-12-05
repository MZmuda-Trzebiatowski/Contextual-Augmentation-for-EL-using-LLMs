SYSTEM_PROMPT = """You are an expert Entity Linking system.
For each entity marked with [START_ENT] and [END_ENT] tags, find the correct Wikipedia URL.
Consider the full context of the sentence to disambiguate entities.
Return a JSON object with a "tags" key containing linked entities."""


def get_user_prompt(nerful_text: str) -> str:
    """
    Generate the user prompt for entity linking.
    
    Args:
        nerful_text: Text with entities tagged using [START_ENT] and [END_ENT].
        
    Returns:
        The formatted user prompt string.
    """
    return f"""Link each tagged entity to its Wikipedia article:

"{nerful_text}"

Return JSON with:
{{"tags": [{{"text": "entity_text", "uri": "https://en.wikipedia.org/wiki/..."}}]}}"""

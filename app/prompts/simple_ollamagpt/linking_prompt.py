JSON_STRUCT = "{text:<tagged_text_AS_IS>, uri:<wikipedia url linking tagged text to the most probable entity given the sentence context>}"

EXAMPLE_OUTPUT = '{tags: [{text: "Angelina", uri:"https://en.wikipedia.org/wiki/Angelina_Jolie"}, {text: "Jon", uri: "https://en.wikipedia.org/wiki/Jon_Voight"}, {text: "Brad", uri: "https://en.wikipedia.org/wiki/Brad_Pitt"}]}'


def get_prompt(nerful_text: str) -> str:
    """
    Generate the entity linking prompt for NER-tagged text.
    
    Args:
        nerful_text: Text with entities tagged using [START_ENT] and [END_ENT].
        
    Returns:
        The formatted prompt string.
    """
    return f"""
Keeping in mind the entire context of the sentence, for each entity tagged with [START_ENT] and [END_ENT] tags in this sentence:
'{nerful_text}'
Generate a tag json object of the following structure:
{JSON_STRUCT}
Return a json object with a list of these tags as the 'tags' key.
Examples:
- 'Angelina, her father Jon, and her partner Brad never played together in the same movie.' -> {EXAMPLE_OUTPUT}
"""

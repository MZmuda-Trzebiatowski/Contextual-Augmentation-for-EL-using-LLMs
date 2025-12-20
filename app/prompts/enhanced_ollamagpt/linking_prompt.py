SYSTEM_PROMPT = """
You are an expert Entity Linking system.
The text you will receive contains tagged entities in the form [START_ENT]<entity>[END_ENT][<list of matching urls>]
For each entity marked with these tags, select the most fitting url from the list.
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
    json_struct = (
        "{text:<tagged_text_AS_IS>, uri:<wikipedia url linking tagged text to the most probable entity given the sentence context>}"
    )

    tag_correct = '{tags: [{text: "Angelina", uri:"https://en.wikipedia.org/wiki/Angelina_Jolie"}, {text: "Jon", uri: "https://en.wikipedia.org/wiki/Jon_Voight"}, {text: "Brad", uri: "https://en.wikipedia.org/wiki/Brad_Pitt"}]}'

    return f"""
            Keeping in mind the entire context of the sentence, for each entity tagged with [START_ENT] and [END_ENT] tags in this sentence:
            '{nerful_text}'
            Generate a tag json object of the following structure:
            {json_struct}
            Return a json object with a list of these tags as the 'tags' key.
            Examples:
            - 'Angelina, her father Jon, and her partner Brad never played together in the same movie.' -> {tag_correct}
            """

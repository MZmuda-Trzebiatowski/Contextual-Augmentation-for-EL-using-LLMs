def get_prompt(text: str) -> str:
    """
    Generate the NER prompt for a given text.
    
    Args:
        text: The input text to perform NER on.
        
    Returns:
        The formatted prompt string.
    """
    return f"""
For the given sentence:
'{text}'
Generate text with named entities surrounded by [START_ENT] and [END_ENT] tags.
Tag ONLY entities likely to represent people, companies, brands, organizations, news outlets etc.
Exclude common words from tags: e.g. 'The white house ...' -> 'The [START_ENT]white house[END_ENT] ...' not '[START_ENT]The white house[END_ENT] ...'
Return ONLY the same text with the proper tags. e.g 'The white house ...' -> 'The [START_ENT]white house[END_ENT] ...' not 'Here is the tagged text: [START_ENT]The white house[END_ENT] ...'
Examples:
- 'Alice has a dog' -> '[START_ENT]Alice[END_ENT] has a [START_ENT]dog[END_ENT]'
- 'Angelina, her father Jon, and her partner Brad never played together in the same movie.' -> '[START_ENT]Angelina[END_ENT], her father [START_ENT]Jon[END_ENT], and her partner [START_ENT]Brad[END_ENT] never played together in the same movie.'
"""

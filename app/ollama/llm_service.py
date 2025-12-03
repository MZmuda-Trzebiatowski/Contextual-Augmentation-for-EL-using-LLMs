import re
import ollama
from app.ollama.models import ELTagExtend, ELTagList


class OllamaGPT:
    def __init__(self, name: str) -> None:
        self.name = name
        ollama.pull(name)

    def run_linking(self, nerful_text: str) -> list[ELTagExtend]:
        """
        Run entity linking on NER tagged text (see :meth:`OllamaGPT.run_ner`)
        """
        json_struct = "{text:<tagged_text_AS_IS>, uri:<wikipedia url linking tagged text to the most probable entity given the sentence context>}"
        tag_correct = '{tags: [{text: "Angelina", uri:"https://en.wikipedia.org/wiki/Angelina_Jolie"}, {text: "Jon", uri: "https://en.wikipedia.org/wiki/Jon_Voight"}, {text: "Brad", uri: "https://en.wikipedia.org/wiki/Brad_Pitt"}]}'
        message = f"""
                Keeping in mind the entire context of the sentence, for each entity tagged with [START_ENT] and [END_ENT] tags in this sentence:
                '{nerful_text}'
                Generate a tag json object of the following structure:
                {json_struct}
                Return a json object with a list of these tags as the 'tags' key.
                Examples:
                - 'Angelina, her father Jon, and her partner Brad never played together in the same movie.' -> {tag_correct}
                """

        response = ollama.chat(
            model=self.name,
            messages=[
                {
                    'role': 'user',
                    'content': message,
                },
            ],
            format=ELTagList.model_json_schema()
        )
        if response is None:
            raise RuntimeError("Received None as a response from model")
        # Model tagged entities outside of tags, if this repeats try
        # ner_tags = set(
        #     x.replace("[START_ENT]", "").replace("[END_ENT]", "").lower()
        #     for x in re.findall(r"\[START_ENT\].+?\[END_ENT\]", nerful_text)
        # )
        # [tag for tag in ELTagList.model_validate_json(response.message.content).tags if tag.text.lower() in ner_tags]
        filtered_tags = ELTagList.model_validate_json(response.message.content).tags

        raw_text = nerful_text.replace("[START_ENT]", "").replace("[END_ENT]", "").lower()

        result = []
        last_idx = 0
        for tag in filtered_tags:
            start = raw_text.index(tag.text.lower(), last_idx)
            end = start + len(tag.text)
            last_idx = end
            result.append(ELTagExtend(text=tag.text, uri=tag.uri, beginIndex=start, endIndex=end))
        return result

    def run_ner(self, text: str) -> str:
        """
        Expand string with entity tags via the llm
        e.g 'Alice has a dog' -> '[START_ENT]Alice[END_ENT] has a [START_ENT]dog[END_ENT]'
        """
        message = f"""
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
        response = ollama.chat(model=self.name, messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        return response['message']['content'].split("</think>")[-1]

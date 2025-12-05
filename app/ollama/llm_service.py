from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import ollama
from tqdm import tqdm

from app.ollama.models import ELTagExtend, ELTagList
from app.prompts.simple_ollamagpt import ner_prompt, linking_prompt


class OllamaGPT(ABC):
    """Base class for Ollama-based LLM services."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        ollama.pull(name)
    
    @abstractmethod
    def run_ner(self, text: str) -> str:
        """Perform Named Entity Recognition on text."""
        pass
    
    @abstractmethod
    def run_linking(self, nerful_text: str) -> list[ELTagExtend]:
        """Perform Entity Linking on NER-tagged text."""
        pass
    
    @abstractmethod
    def run_batch(
        self,
        texts: list[str],
        max_workers: int = 4,
        show_progress: bool = True
    ) -> list[dict]:
        """Run batch processing on multiple texts."""
        pass


class SimpleOllamaGPT(OllamaGPT):
    """Simple Ollama-based LLM service for NER and Entity Linking."""
    
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def run_linking(self, nerful_text: str) -> list[ELTagExtend]:
        """
        Run entity linking on NER tagged text (see :meth:`OllamaGPT.run_ner`)
        """
        message = linking_prompt.get_prompt(nerful_text)

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
        message = ner_prompt.get_prompt(text)
        response = ollama.chat(model=self.name, messages=[
            {
                'role': 'user',
                'content': message,
            },
        ])
        return response['message']['content'].split("</think>")[-1]

    def run_ner_and_linking(self, text: str) -> list[ELTagExtend]:
        """
        Run NER followed by entity linking on a text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of ELTagExtend objects with entity info and Wikipedia URIs
        """
        ner_output = self.run_ner(text)
        return self.run_linking(ner_output)

    def run_batch(
        self,
        texts: list[str],
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: bool = True
    ) -> list[dict]:
        """
        Run NER and entity linking on a batch of texts using concurrent processing.
        
        Args:
            texts: List of input texts to process
            max_workers: Maximum number of concurrent workers (default: 4)
            progress_callback: Optional callback function(current, total) for progress updates
            show_progress: Whether to show a progress bar (default: True)
            
        Returns:
            List of dicts with keys:
                - text: Original input text
                - ner_output: NER tagged text
                - entities: List of ELTagExtend objects
                - error: Error message if processing failed, None otherwise
        """
        results = [None] * len(texts)
        
        def process_single(idx_text: tuple[int, str]) -> tuple[int, dict]:
            idx, text = idx_text
            try:
                ner_output = self.run_ner(text)
                entities = self.run_linking(ner_output)
                return idx, {
                    "text": text,
                    "ner_output": ner_output,
                    "entities": entities,
                    "error": None
                }
            except Exception as e:
                return idx, {
                    "text": text,
                    "ner_output": None,
                    "entities": [],
                    "error": str(e)
                }
        
        indexed_texts = list(enumerate(texts))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single, item): item for item in indexed_texts}
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(texts), desc="Processing texts")
            
            for future in iterator:
                idx, result = future.result()
                results[idx] = result
                if progress_callback:
                    completed = sum(1 for r in results if r is not None)
                    progress_callback(completed, len(texts))
        
        return results

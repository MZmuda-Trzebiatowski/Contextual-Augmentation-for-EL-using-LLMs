import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal, Optional

import ollama
from tqdm import tqdm

from app.ollama.models import ELTagExtend, ELTagList
from app.ollama.llm_service import OllamaGPT
from app.prompts.enhanced_ollamagpt import (
    ner_system_prompt,
    ner_user_prompt,
    linking_system_prompt,
    linking_user_prompt,
    combined_system_prompt,
    combined_user_prompt,
)


class EnhancedOllamaGPT(OllamaGPT):
    """
    Enhanced Ollama-based LLM service for NER and Entity Linking.
    
    Improvements over base OllamaGPT:
    - Retry logic for transient failures
    - More structured prompts
    - Option for combined NER+EL in single call (fewer API calls)
    - Better error handling and logging
    """
    
    def __init__(
        self, 
        name: str, 
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        super().__init__(name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def _call_with_retry(
        self, 
        call_func: Callable, 
        *args, 
        **kwargs
    ) -> dict:
        """Execute an API call with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return call_func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise last_error
    
    def run_combined_ner_and_linking(self, text: str) -> list[ELTagExtend]:
        """
        Run NER and entity linking in a single LLM call.
        
        This approach is more efficient as it requires only one API call,
        and the model can use context from NER to inform linking decisions.
        
        Args:
            text: Input text to process
            
        Returns:
            List of ELTagExtend objects with entity info and Wikipedia URIs
        """
        response = self._call_with_retry(
            ollama.chat,
            model=self.name,
            messages=[
                {"role": "system", "content": combined_system_prompt},
                {"role": "user", "content": combined_user_prompt(text)}
            ],
            format=ELTagList.model_json_schema()
        )
        
        if response is None:
            raise RuntimeError("Received None response from model")
        
        content = response["message"]["content"]
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        
        filtered_tags = ELTagList.model_validate_json(content).tags
        
        raw_text = text.lower()
        result = []
        last_idx = 0
        
        for tag in filtered_tags:
            try:
                start = raw_text.index(tag.text.lower(), last_idx)
                end = start + len(tag.text)
                last_idx = end
                result.append(ELTagExtend(
                    text=tag.text, 
                    uri=tag.uri, 
                    beginIndex=start, 
                    endIndex=end
                ))
            except ValueError:
                try:
                    start = raw_text.index(tag.text.lower())
                    end = start + len(tag.text)
                    result.append(ELTagExtend(
                        text=tag.text, 
                        uri=tag.uri, 
                        beginIndex=start, 
                        endIndex=end
                    ))
                except ValueError:
                    continue
        
        return result
    
    def run_ner(self, text: str) -> str:
        """
        Perform Named Entity Recognition with improved prompts.
        
        Args:
            text: Input text to process
            
        Returns:
            Text with entities tagged using [START_ENT] and [END_ENT]
        """
        response = self._call_with_retry(
            ollama.chat,
            model=self.name,
            messages=[
                {"role": "system", "content": ner_system_prompt},
                {"role": "user", "content": ner_user_prompt(text)}
            ]
        )
        
        content = response["message"]["content"]
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        
        return content
    
    def run_linking(self, nerful_text: str) -> list[ELTagExtend]:
        """
        Perform Entity Linking on NER-tagged text.
        
        Args:
            nerful_text: Text with [START_ENT] and [END_ENT] tags
            
        Returns:
            List of ELTagExtend objects
        """
        response = self._call_with_retry(
            ollama.chat,
            model=self.name,
            messages=[
                {"role": "system", "content": linking_system_prompt},
                {"role": "user", "content": linking_user_prompt(nerful_text)}
            ],
            format=ELTagList.model_json_schema()
        )
        
        if response is None:
            raise RuntimeError("Received None response from model")
        
        content = response["message"]["content"]
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
        
        filtered_tags = ELTagList.model_validate_json(content).tags
        
        raw_text = nerful_text.replace("[START_ENT]", "").replace("[END_ENT]", "").lower()
        
        result = []
        last_idx = 0
        
        for tag in filtered_tags:
            try:
                start = raw_text.index(tag.text.lower(), last_idx)
                end = start + len(tag.text)
                last_idx = end
                result.append(ELTagExtend(
                    text=tag.text, 
                    uri=tag.uri, 
                    beginIndex=start, 
                    endIndex=end
                ))
            except ValueError:
                continue
        
        return result
    
    def run_batch(
        self,
        texts: list[str],
        max_workers: int = 4,
        mode: Literal["combined", "separate"] = "combined",
        show_progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[dict]:
        """
        Run batch processing on multiple texts.
        
        Args:
            texts: List of texts to process
            max_workers: Number of concurrent workers
            mode: "combined" for single-call NER+EL, "separate" for two-stage
            show_progress: Whether to show progress bar
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of result dictionaries
        """
        results = [None] * len(texts)
        
        def process_single(idx_text: tuple[int, str]) -> tuple[int, dict]:
            idx, text = idx_text
            try:
                if mode == "combined":
                    entities = self.run_combined_ner_and_linking(text)
                    ner_output = None
                else:
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

from app.prompts.simple_ollamagpt import ner_prompt, linking_prompt
from app.prompts.enhanced_ollamagpt import (
    ner_system_prompt,
    ner_user_prompt,
    linking_system_prompt,
    linking_user_prompt,
    combined_system_prompt,
    combined_user_prompt,
)

__all__ = [
    # Simple OllamaGPT prompts
    "ner_prompt",
    "linking_prompt",
    # Enhanced OllamaGPT prompts
    "ner_system_prompt",
    "ner_user_prompt",
    "linking_system_prompt",
    "linking_user_prompt",
    "combined_system_prompt",
    "combined_user_prompt",
]

from app.prompts.enhanced_ollamagpt import ner_prompt, linking_prompt, combined_prompt

ner_system_prompt = ner_prompt.SYSTEM_PROMPT
ner_user_prompt = ner_prompt.get_user_prompt

linking_system_prompt = linking_prompt.SYSTEM_PROMPT
linking_user_prompt = linking_prompt.get_user_prompt

combined_system_prompt = combined_prompt.SYSTEM_PROMPT
combined_user_prompt = combined_prompt.get_user_prompt

__all__ = [
    "ner_prompt",
    "linking_prompt", 
    "combined_prompt",
    "ner_system_prompt",
    "ner_user_prompt",
    "linking_system_prompt",
    "linking_user_prompt",
    "combined_system_prompt",
    "combined_user_prompt",
]

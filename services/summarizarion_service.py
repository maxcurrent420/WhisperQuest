import logging
from aimodels.llm_models import get_llm_model
from config import global_state
from memory_profiler import profile

class SummarizationModule:
    @profile
    def __init__(self, llm_selection, groq_api_key=None):
        print(f"Initializing SummarizationModule with {llm_selection}")
        self.llm_model = get_llm_model(llm_selection, groq_api_key)
        print("SummarizationModule initialized")

    @profile
    def summarize_conversation(self, messages):
        print("Summarizing conversation...")
        summary_prompt = {
            'role': 'system',
            'content': 'Please provide a concise summary of the following conversation, focusing on the key events, items, other important details and character traits and actions. Avoid repeating information, Avoid making assumptions or interpretations of the conversation that are not explicitly stated. Respond only with the summary.'
        }
#[TODO- finish summarization module]

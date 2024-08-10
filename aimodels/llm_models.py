import torch
import logging
from openai import OpenAI
from groq import Groq
from config import global_state


class LLMModel:
    def generate_response(self, messages):
        raise NotImplementedError("Subclasses must implement this method")

class LocalLLMModel(LLMModel):
    def __init__(self, base_url="http://localhost:1234/v1", api_key="lm-studio"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate_response(self, messages):
        try:
            with torch.no_grad():
                completion = self.client.chat.completions.create(
                    model="eldogbbhed/3.1",
                    messages=messages,
                    stream=True)
                
                full_response = ""
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        print(content, end="", flush=True)  # Print each chunk as it arrives

                print()  # Print a newline after the full response
                return full_response
        except Exception as e:
            print(f"Error with local LLM: {e}")
            return f"Error: Unable to get response from local LLM. {str(e)}"

class GroqLLMModel(LLMModel):
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate_response(self, messages):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                max_tokens=8000,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error with Groq API: {e}")
            return f"Error: Unable to get response from Groq. {str(e)}"

def get_llm_model(llm_selection, groq_api_key=None):
    print(f"Getting LLM model: {llm_selection}")
    if llm_selection == "Local":
        print("Initializing LocalLLMModel")
        return LocalLLMModel()
    elif llm_selection == "Groq":
        if not groq_api_key:
            print("Error: Groq API key is required for Groq LLM")
            raise ValueError("Groq API key is required for Groq LLM")
        print("Initializing GroqLLMModel")
        return GroqLLMModel(groq_api_key)
    else:
        print(f"Error: Invalid LLM selection: {llm_selection}")
        raise ValueError(f"Invalid LLM selection: {llm_selection}")

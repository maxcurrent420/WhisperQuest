# --- main.py ---
import gradio as gr
from ui.gradio_interface import launch_interface
#from utils.logging_config import setup_logging
from services.story_service import interactive_storyteller
from config import global_state
import os
os.environ["GROQ_API_KEY"] = "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"

def main():
   # setup_logging()
    launch_interface(interactive_storyteller, global_state)

if __name__ == "__main__":
    main()

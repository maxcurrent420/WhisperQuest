# config.py

import os
import torch
# import gradio as gr  # Removed Gradio import
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import time
import logging
import io
import sys
import RealtimeTTS
from RealtimeSTT import AudioToTextRecorder
from TTS.api import TTS
import gc
import re
from openai import OpenAI
from groq import Groq
import requests
import base64
from PIL import Image
import tempfile
import os

# Set environment variable for Groq API key (if you're using Groq)
os.environ["GROQ_API_KEY"] = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # Replace with your actual API key

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
output_dir = 'outputs'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(output_dir, exist_ok=True)

class GlobalState:
    """
    Global state for the application.

    This class holds global configuration settings and objects to prevent unnecessary object holding in functions.
    """
    def __init__(self):
        self.recorder = None
        self.running = True
        self.muted = False
        self.ckpt_converter = '/path/to/checkpoints/converter/here/checkpoints/converter'
        self.en_ckpt_base = '/path/to/checkpoints/base_speakers/EN'
        self.playback_speed = 0.7
        self.unrealspeech_api_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" #Enter API Key Here for Unreal Speech Support   
        self.paused = False
        self.message_queue = []
        self.scenario_type = "Thriller"
        self.xtts_engine = None
        self.output_dir = 'outputs'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yourtts_engine = None
        self.llm_selection = "Groq"
        self.selected_voice_model = "edge_tts"
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.last_llm_response = ""
        self.image_generation_enabled = True  # Default for image generation

        # **Add these lines to ensure GlobalState is not unnecessarily held in functions**
        self.audio_service = None
        self.story_service = None
        self.image_service = None

global_state = GlobalState()

# Supported languages
SUPPORTED_LANGUAGES = ['zh', 'en']  # Supported languages for the application


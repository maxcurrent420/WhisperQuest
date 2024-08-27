# --- services/image_service.py ---
import requests
import logging
import tempfile
import os
from config import global_state
from aimodels.llm_models import get_llm_model
from PIL import Image
import io
from urllib.parse import quote
import streamlit as st #  Import streamlit for image display

class ImageService:
    def __init__(self, llm_selection, api_key):
        self.api_key = global_state.groq_api_key
        self.llm_selection = llm_selection
        self.temp_files = []  # Store temporary file names
        self.current_image = None  # Initialize current image to None
        self.api_url = "https://image.pollinations.ai/prompt/"

    def generate_caption(self, llm_response):
        # Get the appropriate LLM model
        llm_model = get_llm_model(self.llm_selection, self.api_key)

        # Generate a caption using the LLM model
        caption = llm_model.generate_response(
            [
                {'role': 'system', 'content': 'Please provide a concise and descriptive caption for the following scene, using mostly adjectives to describe what it looks like, not just what is happening, making suitable for image generation, avoiding all exposition, storytelling, the past, and words that describe things that are not real such as [realistic, lifelike, animated, or photorealistic], and use words like real photo instead; mention nothig else whatsoever, never leave comments or a note: etc., always responding with the desired caption, adhering to the following format: [Subject] [Action/Scenario], [Adjective 1] [Adjective 2], [Adjective 3], [Style] [Artistic Medium] [Superlative Adjectives (e.g. hyperdetailed, unbelievably convincing, perfect lighting) ] Correct: Rainbow-colored Mushroom Cloud, majestically towering above a ravaged cityscape, psychedelic hues undulating like iridescent oil slicks, vibrant, bizarre, hypnotic, dreamlike, hallucinogenic, surreal, hyper-saturated, real photo, glowing, gritty, frightening, apocalyptic.'},
                {'role': 'user', 'content': llm_response}
            ]
        )

        if isinstance(caption, str):
            return caption.strip()[:1000]  # Limit to 1000 characters
        else:
            logging.error(f"Invalid caption generated: {caption}")
            return None

    def fetch_image(self, caption):
        if not caption:
            logging.error("No prompt provided for image generation.")
            return None

        try:
            # URL encode the caption
            encoded_caption = quote(caption)

            # Set up the request parameters
            params = {
                'model': 'flux',
                'width': 1021,
                'height': 577,
                'nologo': 'true',
                'enhance': 'false'
            }

            # Construct the full URL
            full_url = f"{self.api_url}{encoded_caption}"

            # Make the API request to Pollinations with SSL verification disabled
            response = requests.get(full_url, params=params, verify=False)

            if response.status_code == 200:
                # Save the raw image data to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_file.write(response.content)
                temp_file.close()
                self.temp_files.append(temp_file.name)  # Store the temp file name
                


                return temp_file.name  # Return the path of the saved image file
            else:
                logging.error(f"Failed to generate image: {response.text}")
                return None

        except Exception as e:
            logging.error(f"Error fetching image from Pollinations API: {e}")
            return None

    def cleanup_temporary_files(self):
        """Deletes temporary files created during image generation."""
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
                logging.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logging.error(f"Error deleting temporary file {temp_file}: {e}")

        self.temp_files.clear()

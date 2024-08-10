import logging
import requests
import base64
import io
import tempfile
from PIL import Image

class ImageService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://api.segmind.com/v1/flux-schnell"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": "SG_d50743ab27d82518"
        }

    def generate_caption(self, llm_response):
        caption_prompt = {
            'role': 'system',
            'content': 'Please provide a concise and descriptive caption for the following scene, using mostly adjectives to describe what it looks like, not just what is happening, making suitable for image generation, avoiding all exposition, storytelling, the past, and words that describe things that are not real such as [realistic, lifelike, animated, or photorealistic], and use words like real photo instead; mention nothig else whatsoever, never leave comments or a note: etc., always responding with the desired caption, adhering to the following format: [Subject] [Action/Scenario], [Adjective 1] [Adjective 2], [Adjective 3], [Style] [Artistic Medium] [Superlative Adjectives (e.g. hyperdetailed, unbelievably convincing, perfect lighting) ] Correct: Rainbow-colored Mushroom Cloud, majestically towering above a ravaged cityscape, psychedelic hues undulating like iridescent oil slicks, vibrant, bizarre, hypnotic, dreamlike, hallucinogenic, surreal, hyper-saturated, real photo, glowing, gritty, frightening, apocalyptic.'
        }
        
        messages = [
            caption_prompt,
            {'role': 'user', 'content': llm_response}
        ]
        
        # Note: This method assumes you have a way to generate a response from an LLM.
        # You might need to import and use your LLM model here.
        caption = self.generate_llm_response(messages)
        
        if isinstance(caption, str):
            return caption.strip()[:200]  # Limit to 200 characters
        else:
            logging.error(f"Invalid caption generated: {caption}")
            return None

    def generate_llm_response(self, messages):
        # Implement this method to generate a response from your LLM
        # This is a placeholder and should be replaced with actual LLM logic
        pass

    def fetch_image_from_caption(self, caption):
        if not caption or not isinstance(caption, str):
            logging.error("Invalid caption: must be a non-empty string")
            return None

        try:
            payload = {
                "prompt": caption,
                "negative_prompt": "unrealistic, unconvincing, deformed, blurry, distorted",
                "samples": 1,
                "scheduler": "DPM++ SDE",
                "num_inference_steps": 25,
                "guidance_scale": 1,
                "seed": -1,
                "img_width": 1024,
                "img_height": 1024,
                "base64": True
            }
            
            response = requests.post(self.url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            image_base64 = response.json().get('image')
            if not image_base64:
                logging.error("Failed to get image from Segmind API response.")
                return None

            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                image.save(temp_file, format='PNG')
                temp_file_path = temp_file.name

            return temp_file_path

        except requests.RequestException as e:
            logging.error(f"Error making image request: {e}")
            return None

    def generate_image(self, llm_response):
        caption = self.generate_caption(llm_response)
        if caption:
            return self.fetch_image_from_caption(caption)
        return None

def cleanup_temporary_files():
    # This function should be called when the application is shutting down
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(temp_dir, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                logging.error(f"Error deleting temporary file {file_path}: {e}")

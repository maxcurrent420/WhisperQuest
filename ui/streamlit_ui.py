import streamlit as st
import os
from services.story_service import StoryService
from services.story_service import interactive_storyteller
from config import global_state
from services.audio_service import AudioService
from services.image_service import ImageService
import tempfile
from PIL import Image
import time
import asyncio
import edge_tts

def launch_interface(story_function, global_state):
    st.title("Verbal Storyteller")

    # Scenario Selection
    scenario_options = ["Thriller", "Fantasy", "Sci-Fi", "Mystery", "Custom"]
    scenario = st.selectbox("Choose a scenario:", scenario_options)
    user_custom_scenario = ""
    if scenario == "Custom":
        user_custom_scenario = st.text_input("Enter your custom scenario:")

    # Voice Model Selection
    voice_model_options = ["edge_tts", "OpenVoice", "xttsv2", "yourtts", "UnrealSpeech"]
    voice_model = st.selectbox("Select a voice model:", voice_model_options)

    # LLM Source Selection
    llm_source = st.radio("Choose LLM Source", ("Local", "Groq"))
    global_state.llm_selection = llm_source

    # Reference Audio Upload (only for OpenVoice)
    reference_audio = None
    if voice_model == "OpenVoice":
        reference_audio = st.file_uploader("Upload reference audio (optional):", type=["wav", "mp3"])

    # Voice Style Selection
    voice_style_options = ['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']
    voice_style = st.selectbox("Select a voice style:", voice_style_options)

    # Enable Image Generation
    enable_image_generation = st.checkbox("Enable Image Generation", value=False)

    # Start Button
    start_button = st.button("Start Storytelling")

    # Story Output - Use st.empty() for dynamic updates
    story_output = st.empty()

    # Image Display Area
    image_placeholder = st.empty()

    # Feedback/Error Messages Display
    feedback_placeholder = st.empty()

    # Initialize services
    audio_service = AudioService(
        global_state.en_ckpt_base, global_state.ckpt_converter, global_state.output_dir
    )
    global_state.audio_service = audio_service

    image_service = None
    if enable_image_generation:
        image_service = ImageService(global_state.groq_api_key)
        global_state.image_service = image_service

    global_state.story_service = StoryService(global_state.llm_selection, global_state.groq_api_key)

    # Loop Control
    if start_button:
        # Set the indicator to "Thinking..."
        feedback_placeholder.text("Thinking...")
        story_output.empty()
        image_placeholder.empty()
        
        # Start the story generation
        # Ensure the audio service is initialized and loaded
        audio_service.set_model(voice_model)

        # Update Global State
        global_state.selected_voice_model = voice_model
        global_state.scenario_type = scenario if scenario != "Custom" else user_custom_scenario
        global_state.image_generation_enabled = enable_image_generation

        story_generator = story_function(
            reference_audio,
            voice_style,
            voice_model,
            scenario,
            user_custom_scenario,
            enable_image_generation,
        )

        # Initialize story_content to store the entire story
        story_content = ""

        # Handle the initial message and subsequent responses
        try:
            while True:
                user_input, story_response, audio_path, indicator, image_path = next(story_generator)

                # Update UI components only if there's new content
                if user_input:
                    story_content += f"\n\nUser: {user_input}\n"
                if story_response:
                    story_content += f"\n{story_response}\n"
                    story_output.markdown(story_content)  # Update story display before audio playback

                feedback_placeholder.text(indicator)

                if image_path:
                    try:
                        image = Image.open(image_path)
                        image_placeholder.image(image)
                    except FileNotFoundError:
                        image_placeholder.text("Error loading image.")

                # Audio playback would occur here (not implemented in this UI code)

                time.sleep(0.1)

        except StopIteration:
            # Handle the case where the generator is exhausted
            feedback_placeholder.text("Story has ended.")

        # Clean up audio and image services if they were used.
        if audio_service:
            audio_service.cleanup()
        if image_service:
            image_service.cleanup_temporary_files()

    # User input (microphone)
    with st.sidebar:
        st.subheader("Speak to the Storyteller")
        st.text("Say 'start' to begin, and choose your actions as the story unfolds.")

if __name__ == "__main__":
    launch_interface(interactive_storyteller, global_state)

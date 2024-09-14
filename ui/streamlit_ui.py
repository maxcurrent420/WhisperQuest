# --- ui/streamlit_ui.py ---

import streamlit as st
import os
from services.story_service import StoryService
from services.story_service import interactive_storyteller
from config import global_state
from services.audio_service import AudioService
import tempfile
from PIL import Image
import time
import streamlit as st
from services.image_service import ImageService  # Import the ImageService
from memory_profiler import profile

def launch_interface(story_function, global_state):
    st.set_page_config(
        page_title="WhisperQuest",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={'About': "# WhisperQuest\nAn interactive storytelling experience."},
    )
    # Add custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #111827; 
            color: #f5f5f5;
            font-family: 'Times New Roman', sans-serif;
        }
        .custom-header {
            background-color: #0F5005; /* Blue background */
            padding: 10px;
            color: white;
            border-radius: 5px; /* Rounded corners */
        }
        </style>
        """, unsafe_allow_html=True)

    # Apply custom header styling
    st.markdown('<h1 class="custom-header">🎭 WhisperQuest</h1>', unsafe_allow_html=True)
    st.subheader("Your adventure awaits, limited only by your imagination and token context!")

    # Scenario Selection
    scenario_options = ["Star Quest", "Max Hammer", "Custom", "Thriller", "Fantasy", "Sci-Fi", "Mystery"]
    scenario = st.selectbox("Choose a scenario:", scenario_options, key="scenario")
    user_custom_scenario = ""
    if scenario == "Custom":
        user_custom_scenario = st.text_input("Enter your custom scenario:", key="custom_scenario")

    # Voice Model Selection
    voice_model_options = ["edge_tts", "MeloTTS", "OpenVoiceV2", "xttsv2", "yourtts", "UnrealSpeech"]
    voice_model = st.selectbox("Select a voice model:", voice_model_options, key="voice_model")

    # LLM Source Selection
    llm_source = st.radio("Choose LLM Source", ("Groq", "Local", "Gemini"), key="llm_source")
    global_state.llm_selection = llm_source

    # Reference Audio Upload
    reference_audio = None
    supported_engines = {
    "OpenVoiceV2": True,
    "yourtts": True,
    "xttsv2": True  # Add more engines as needed
}

    if voice_model in supported_engines and supported_engines[voice_model]:
        playback_speed = st.slider("Playback Speed", min_value=0.1, max_value=1.5, value=float(global_state.playback_speed), step=0.1, key="playback_speed")
        global_state.playback_speed = playback_speed
        reference_audio = st.file_uploader("Upload reference audio (optional):", type=["wav", "mp3"], key="reference_audio")

    # Voice Style Selection
    voice_style_options = ['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']
    voice_style = st.selectbox("Select a voice style:", voice_style_options, key="voice_style")

    # Start Button
    start_button = st.button("Start Storytelling", key="start_story")

    # Story Output
    story_output = st.empty()
    feedback_placeholder = st.empty()
    image_placeholder = st.empty()  # Placeholder for the image

    # Initialize services
    audio_service = AudioService(
    en_ckpt_base=global_state.en_ckpt_base,
    ckpt_converter=global_state.ckpt_converter,
    en_ckpt_base_v2=global_state.en_ckpt_base_v2,
    ckpt_converter_v2=global_state.ckpt_converter_v2,
    output_dir=global_state.output_dir
)
    global_state.audio_service = audio_service
    global_state.story_service = StoryService(global_state.llm_selection, global_state.groq_api_key, global_state.gemini_api_key)
    image_service = ImageService(global_state.llm_selection, global_state.gemini_api_key,)

    # Loop Control
    if start_button:
        feedback_placeholder.text("Thinking...")
        story_output.empty()
        image_placeholder.empty()  # Clear the image placeholder

        # Start the story generation
        audio_service.set_model(voice_model)
        global_state.selected_voice_model = voice_model
        global_state.scenario_type = scenario if scenario != "Custom" else user_custom_scenario

        story_generator = story_function(
            reference_audio,
            voice_style,
            voice_model,
            scenario,
            user_custom_scenario,
        )

        # Initialize story_content
        story_content = ""

        try:
            while True:
                user_input, story_response, audio_path, indicator, image_path = next(story_generator)

                if user_input:
                    story_content += f"\n\nUser: {user_input}\n"
                if story_response:
                    story_content += f"\n{story_response}\n"
                    story_output.markdown(story_content)

                if image_path:
                    try:
                        image = Image.open(image_path)
                        image_placeholder.image(image, caption="Generated Image", use_column_width=True)
                    except Exception as e:
                        print(f"Error displaying image: {e}")

                feedback_placeholder.text(indicator)

                time.sleep(0.1)

        except StopIteration:
            feedback_placeholder.text("Story has ended.")
        #audio_service.cleanup()  # Moved to after the loop
        if image_service:
            image_service.cleanup_temporary_files()
        if audio_service:
            audio_service.cleanup()

    with st.sidebar:
        st.subheader("Speak to the Storyteller")
        st.text("Say 'start' to begin, and choose your actions as the story unfolds.")

if __name__ == "__main__":
    launch_interface(interactive_storyteller, global_state)

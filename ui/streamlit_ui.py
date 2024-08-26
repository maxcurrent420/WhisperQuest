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
    st.subheader("Embark on an interactive storytelling journey limited only by your imagination and token context!")

    # Scenario Selection
    scenario_options = ["Star Quest", "Clay Hammer", "Custom", "Thriller", "Fantasy", "Sci-Fi", "Mystery"]
    scenario = st.selectbox("Choose a scenario:", scenario_options, key="scenario")
    user_custom_scenario = ""
    if scenario == "Custom":
        user_custom_scenario = st.text_input("Enter your custom scenario:", key="custom_scenario")

    # Voice Model Selection
    voice_model_options = ["edge_tts", "OpenVoice", "xttsv2", "yourtts", "UnrealSpeech"]
    voice_model = st.selectbox("Select a voice model:", voice_model_options, key="voice_model")

    # LLM Source Selection
    llm_source = st.radio("Choose LLM Source", ("Groq", "Local"), key="llm_source")
    global_state.llm_selection = llm_source

    # Reference Audio Upload
    reference_audio = None
    if voice_model == "OpenVoice":
        reference_audio = st.file_uploader("Upload reference audio (optional):", type=["wav", "mp3"], key="reference_audio")

    # Voice Style Selection
    voice_style_options = ['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']
    voice_style = st.selectbox("Select a voice style:", voice_style_options, key="voice_style")

    # Start Button
    start_button = st.button("Start Storytelling", key="start_story")

    # Story Output
    story_output = st.empty()
    feedback_placeholder = st.empty()

    # Initialize services
    audio_service = AudioService(
        global_state.en_ckpt_base, global_state.ckpt_converter, global_state.output_dir
    )
    global_state.audio_service = audio_service
    global_state.story_service = StoryService(global_state.llm_selection, global_state.groq_api_key)

    # Loop Control
    if start_button:
        feedback_placeholder.text("Thinking...")
        story_output.empty()
        
        # Scenario image display
        if scenario == "Clay Hammer":
            scenario_image = Image.open("./images/Clay_Hammer.jpg")
            st.image(scenario_image, caption="Detective Clay Hammer", width=672)
        elif scenario == "Star Quest":
            scenario_image = Image.open("./images/Star_Quest.jpg")
            st.image(scenario_image, caption="The Starship Whisperion", width=672)

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
                user_input, story_response, audio_path, indicator, _ = next(story_generator)  # Added "_" to unpack the extra value

                if user_input:
                    story_content += f"\n\nUser: {user_input}\n"
                if story_response:
                    story_content += f"\n{story_response}\n"
                    story_output.markdown(story_content)

                feedback_placeholder.text(indicator)

                time.sleep(0.1)

        except StopIteration:
            feedback_placeholder.text("Story has ended.")
        if audio_service:
            audio_service.cleanup()




if __name__ == "__main__":
    launch_interface(interactive_storyteller, global_state)

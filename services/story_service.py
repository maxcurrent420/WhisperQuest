# --- services/story_service.py ---
import logging
from aimodels.llm_models import get_llm_model
from config import global_state
from services.audio_service import AudioService, EnhancedAudioToTextRecorder
import gc
from memory_profiler import profile


class StoryService:
    @profile
    def __init__(self, llm_selection, groq_api_key=None):
        print(f"Initializing StoryService with {llm_selection}")
        self.llm_model = get_llm_model(llm_selection, groq_api_key)
        self.messages = []
        print("StoryService initialized")

    @profile
    def set_scenario(self, scenario_type):
        print(f"Setting scenario: {scenario_type}")
        system_prompt = {
            'role': 'system',
            'content': f'''You are an interactive verbal storyteller/Narrator specializing in narratives about {scenario_type}. Your role is to create an immersive , intense, funny and suspenseful experience for the player. 
            Begin by welcoming the player and introducing the {scenario_type} genre as a freandly narrator. Then, set up the initial scenario where the player finds themselves in a thrilling situation. 
            Provide three actions for the player to choose from at each decision point. Adapt the story based on their choices, maintaining suspense and intrigue throughout the narrative. 
            Be creative, unpredictable, and engage the player's imagination with plot twists, and intense intrigue and action. Ensure every response ends with a new set of three new choices for the player and advances the story with a sense of realism and grit. 
            Provide more challenging choices as the story progresses, requiring users to think more strategically about their consequences. 
            More complex choices should lead to even more challenging story arcs. 
            Also remember that the user may be using speech to text software and responses may be garbled, so if their choice seems unintelligible, then just ask "Sorry, I didn't catch that" and then repeat the three options. 
            Allow the user to make up their own selection, rejecting all the given choices if they choose. Do not argue with the player. 
            Do not refuse to accept their choices. Do not add disclaimers, warnings or other text not directly related to the story. 
            Maintain the immersive experience. 
            When the player says start.'''
        }
        self.messages = [system_prompt]
        print("Scenario set")

    @profile
    def generate_initial_message(self):
        print("Generating initial message")
        initial_message = "Welcome to the Infinite A I ReActive Experience. Are you ready to begin your adventure? Say 'Start' when you're ready."
        self.messages.append({"role": "assistant", "content": initial_message})
        print(f"Initial message: {initial_message}")
        return initial_message

    @profile
    def generate_response(self, user_input):
        print(f"Generating response for user input: {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        
        print("Calling LLM model")
        try:
            story_response = self.llm_model.generate_response(self.messages)
            
            if story_response:
                print(f"Received response: {story_response[:50]}...")  # Print first 50 characters
                self.messages.append({"role": "assistant", "content": story_response})
                return story_response
            else:
                print("Failed to generate story response")
                return "Error: Failed to generate story response"
        except Exception as e:
            print(f"Error in generate_response: {e}")
            return f"Error: Failed to generate story response. {str(e)}"


@profile
def interactive_storyteller(reference_audio, voice_style, voice_model, scenario, user_custom_scenario, enable_image_generation):
    """
    Main function for the interactive story teller.
    """
    print("Starting interactive_storyteller")
    global_state.selected_voice_model = voice_model
    global_state.scenario_type = scenario if scenario != "Custom" else user_custom_scenario
    global_state.image_generation_enabled = enable_image_generation

    print(f"LLM selection: {global_state.llm_selection}")
    print(f"Using voice model: {global_state.selected_voice_model}")
    print(f"Image generation enabled: {global_state.image_generation_enabled}")

    # Use services from global_state
    audio_service = global_state.audio_service
    story_service = global_state.story_service
    image_service = global_state.image_service if global_state.image_generation_enabled else None
    
    story_service.set_scenario(global_state.scenario_type)
    print("Generating initial message")
    initial_message = story_service.generate_initial_message()
    print("Generating audio for initial message")
    audio_path = audio_service.generate_and_play_audio(initial_message, reference_audio, voice_style)

    print("Yielding initial message")
    yield "", initial_message, audio_path, "Speak now", None

    # Initialize output variables
    user_input = ""
    story_output = initial_message
    audio_output = audio_path
    indicator = "Speak now"
    image_output = None
    global_state.recorder = EnhancedAudioToTextRecorder(
            model="tiny",
            language="en",
            spinner=True,
            wake_word_activation_delay=5
        )


    while True:
        # Load the selected voice model at the beginning of the loop
        audio_service.set_model(global_state.selected_voice_model) 

        if global_state.paused:
            time.sleep(1)
            continue


        user_input = global_state.recorder.text()
        global_state.recorder.stop()
        global_state.recorder.reset()
        global_state.recorder.clear_audio_queue()

        if user_input.lower() == "quit":
            break

        # Update the indicator to "Thinking"
        yield user_input, "Thinking...", audio_path, "Thinking", None

        print(f"Generating response for user input: {user_input}")
        story_response = story_service.generate_response(user_input)

        if story_response:
            print(f"Received response: {story_response[:50]}...")  # Print first 50 characters
            print("Generating audio for story response")
            audio_path = audio_service.generate_and_play_audio(story_response, reference_audio, voice_style)

            image_path = None
            if image_service:
                caption = image_service.generate_caption(story_response)
                if caption:
                    image_path = image_service.fetch_image_from_caption(caption)
                
                if image_path:
                    print(f"Image fetched successfully: {image_path}")
                else:
                    print("Failed to fetch image.")
            else:
                print("Image generation is disabled, skipping caption and image generation")

            # Update UI Components
            # user_input.value = user_input (This is incorrect as user_input is a string)
            story_output = story_output + story_response  # Append to story_output
            audio_output = audio_path
            indicator = "Speaking"
            image_output = image_path
            
            # **Crucial Change: Yield the updated values**
            yield user_input, story_output, audio_path, "Speaking", image_path

        else:
            print("Failed to generate story response")
            #yield user_input, "Error: Failed to generate story response", None, "Error", None
            
            # user_input.value = user_input (This is incorrect as user_input is a string)
            story_output = "Error: Failed to generate story response"
            audio_output = None
            indicator = "Error"
            image_output = None

            # **Crucial Change: Yield the updated values**
            yield user_input, story_output, None, "Error", None

        # Clean up temporary files
        if image_service:
            image_service.cleanup_temporary_files()

        # Perform garbage collection
        gc.collect() 

    # Clean up audio service
    audio_service.cleanup()
    print("interactive_storyteller finished")

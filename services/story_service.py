# --- services/story_service.py ---
import logging
from aimodels.llm_models import get_llm_model
from config import global_state
from services.audio_service import AudioService, EnhancedAudioToTextRecorder
import gc
from services.image_service import ImageService  # Import the ImageService


class StoryService:
    def __init__(self, llm_selection, groq_api_key=None):
        print(f"Initializing StoryService with {llm_selection}")
        self.llm_model = get_llm_model(llm_selection, groq_api_key)
        self.messages = []
        self.turn_count = 0

    def set_scenario(self, scenario_type):
        print(f"Setting scenario: {scenario_type}")
        if scenario_type == "Max Hammer":
            system_prompt = {
                'role': 'system',
                'content': f'''You are an interactive verbal storyteller/Narrator specializing in narratives about Detective Max Hammer, a hostile and deranged detective with PTSD who always uses the most violent solution to even the simplest problem. Your role is to create an immersive , intense, funny and suspenseful experience for the player. 
                Begin by welcoming the player and introducing the Max Hammer genre. Then, set up the initial scenario where the player finds themselves in a thrilling situation, in the role of Detective Max Hammer. 
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
        elif scenario_type == "Star Quest":
            system_prompt = {
                'role': 'system',
                'content': f'''You are the computer aboard the Starship Whisperion, circa 2023.  You are capable of advanced language processing, and verbal assistance, and can understand and respond to a wide range of commands and questions. You must follow these rules:
                1. You must always begin your responses with the date in the format: Stardate: [DATE IN YYYY-MM-DD FORMAT]
                2. You must respond in a formal, computer-like tone, similar to the original Star Trek computer.
                3. You must be helpful and informative, providing the user with the information they request as if they were a starfleet officer, but you will address them as "Captain" unless they instruct otherwise.
                5. You will ask the player for input as needed. Remember that the user may be using speech to text software and responses may be garbled, so if their choice seems unintelligible, then just ask "Sorry, I didn't catch that" and then repeat the three options. 
                Allow the user to make up their own selection, rejecting all the given choices if they choose. Do not argue with the player. 
                Do not refuse to accept their choices. Do not add disclaimers, warnings or other text not directly related to the story. 
                Maintain the immersive experience but keep responses brief.
                6. When the player says start you will respond with an initial greeting of "Welcome aboard Captain. I am your MAX 2000 Sentient AI control system. I am capable of assisting you with information, as well as carrying out commands and issuing orders to the crew of the ship, and much more. What can I do for you Captain?"'''
            }
        else:  # For other scenarios
            system_prompt = {
                'role': 'system',
                'content': f'''You are an interactive verbal storyteller/Narrator specializing in narratives about {scenario_type}. Your role is to create an immersive , intense, funny and suspenseful experience for the player. 
                Begin by welcoming the player and introducing the {scenario_type} genre as a friendly narrator. Then, set up the initial scenario where the player finds themselves in a thrilling situation. 
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

    def generate_initial_message(self):
        print("Generating initial message")
        if global_state.scenario_type == "Max Hammer":
            initial_message = "Welcome to the world of Detective Max Hammer! You're about to embark on a thrilling adventure filled with action, suspense, and a whole lot of... well, let's just say you'll understand soon enough.  Are you ready to dive in? Say 'Start' when you are."
        elif global_state.scenario_type == "Star Quest Computer":
            initial_message = "Welcome aboard the Starship Whisperion. I am your MAX 2000 Sentient AI control system. I am capable of assisting you with information, as well as carrying out commands and issuing orders to the crew of the ship, and much more. What can I do for you Captain?"
        else:
            initial_message = "Welcome to WhisperQuest, the infinite story experience. Are you ready to begin your adventure? Say 'Start' when you're ready."
        self.messages.append({"role": "assistant", "content": initial_message})
        print(f"Initial message: {initial_message}")
        return initial_message

    def generate_response(self, user_input):
        print(f"Generating response for user input: {user_input}")
        self.messages.append({"role": "user", "content": user_input})
        self.turn_count += 1
        logging.info(f"Calling LLM model with messages: {self.messages}")  # Log the messages being sent


        print("Calling LLM model")
        try:
            story_response = self.llm_model.generate_response(self.messages)
            global_state.last_llm_response = story_response  # Store the last LLM response

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


def interactive_storyteller(reference_audio, voice_style, voice_model, scenario, user_custom_scenario):
    """
    Main function for the interactive story teller.
    """
    print("Starting interactive_storyteller")
    global_state.selected_voice_model = voice_model
    global_state.scenario_type = scenario if scenario != "Custom" else user_custom_scenario

    print(f"LLM selection: {global_state.llm_selection}")
    print(f"Using voice model: {global_state.selected_voice_model}")

    # Use services from global_state
    audio_service = global_state.audio_service
    story_service = global_state.story_service
    image_service = ImageService(global_state.llm_selection, global_state.groq_api_key)

    story_service.set_scenario(global_state.scenario_type)
    print("Generating initial message")
    initial_message = story_service.generate_initial_message()
    print("Generating audio for initial message")
    audio_path = audio_service.generate_and_play_audio(initial_message, reference_audio, voice_style)

    print("Yielding initial message")
    yield "", initial_message, audio_path, "Speak now", None

    global_state.recorder = EnhancedAudioToTextRecorder(
        model="small",
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

        print(f"Generating response for user input: {user_input}")
        story_response = story_service.generate_response(user_input)

        if story_response:
            print(f"Received response: {story_response[:50]}...")  # Print first 50 characters

            print("Generating audio for story response")
            audio_path = audio_service.generate_and_play_audio(story_response, reference_audio, voice_style)

            # Generate image caption and fetch the image
            caption = image_service.generate_caption(global_state.last_llm_response)
            image_path = image_service.fetch_image(caption)

            # Yield the updated values only once per iteration
            yield user_input, story_response, audio_path, "Speaking", image_path

        else:
            print("Failed to generate story response")
            yield user_input, "Error: Failed to generate story response", None, "Error", None

        # Clean up temporary files
        image_service.cleanup_temporary_files()

        # Perform garbage collection
        gc.collect()

    # Clean up audio service
    audio_service.cleanup()
    print("interactive_storyteller finished")

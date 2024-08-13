import torch
import os
import gradio as gr
from services.audio_service import AudioService, EnhancedAudioToTextRecorder
from services.story_service import interactive_storyteller
from services.image_service import ImageService
import logging
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from config import global_state  # Import the global_state instance
from memory_profiler import profile

@profile
def launch_interface(interactive_storyteller_func, global_state):
    en_ckpt_base = '[Enter your full path to openvoice models/for/voice/cloning/support]'
    ckpt_converter = '/enter/your/path/to/openvoice/checkpoints/converter'
    output_dir = 'outputs'
    
    # **Create AudioService here and store it in global_state**
    global_state.audio_service = AudioService(en_ckpt_base=global_state.en_ckpt_base, 
                             ckpt_converter=global_state.ckpt_converter,
                             output_dir=output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)

    en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
    en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    en_source_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
    from services.story_service import StoryService  # Import here to avoid circular imports
    # **Create StoryService here and store it in global_state**
    #global_state.story_service = StoryService(global_state.llm_selection, global_state.groq_api_key)

    # **Create ImageService here (if needed) and store it in global_state**
    if global_state.image_generation_enabled:
        global_state.image_service = ImageService(global_state.groq_api_key)

    with gr.Blocks() as demo:
        gr.Markdown("# Verbal AI Powered Choose Your Own Adventure Game!")
        
        with gr.Row():
            ref_audio = gr.Audio(label="Reference Audio", type="filepath")
            style = gr.Dropdown(label="Voice Style", choices=['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'], value="default")
            voice_model = gr.Dropdown(label="Voice Model", choices=['OpenVoice', 'xttsv2', 'yourtts','UnrealSpeech'], value="OpenVoice")
            llm_choice = gr.Radio(label="LLM Source", choices=["Local", "Groq"], value="Local")
            current_llm = gr.Textbox(label="Current LLM", value="Local", interactive=False)
            scenario = gr.Dropdown(label="Scenario", choices=['Thriller', 'Custom'], value="Thriller")
            custom_scenario = gr.Textbox(label="Custom Scenario", placeholder="Describe your custom scenario here...", interactive=True)
            enable_image_generation = gr.Checkbox(label="Enable Image Generation", value=False)

        start_button = gr.Button("Start Story")
        mute_button = gr.Button("Mute/Unmute")
        pause_button = gr.Button("Pause/Resume")
        print(f"Current voice model: {global_state.selected_voice_model}")

        
        user_input = gr.Textbox(label="Your last input")
        story_output = gr.Textbox(label="Story")
        audio_output = gr.Audio(label="Generated Audio")
        indicator = gr.Textbox(label="Status Indicator", interactive=False)
        image_output = gr.Image(label="Scene Image")
        
        def update_voice_model(new_model):
            global_state.selected_voice_model = new_model
            global_state.audio_service.set_model(new_model)
            print(f"Voice model updated to {new_model}")
            return f"Voice model updated to {new_model}"

        def toggle_mute():
            global_state.muted = not global_state.muted
            return "Unmute" if global_state.muted else "Mute"

        def toggle_pause():
            global_state.paused = not global_state.paused
            return "Resume" if global_state.paused else "Pause"

        def update_llm_selection(new_selection):
            global_state.llm_selection = new_selection
            print(f"LLM selection updated to {new_selection}")
            return new_selection
        
        llm_choice.change(update_llm_selection, inputs=[llm_choice], outputs=[current_llm])
        mute_button.click(toggle_mute, outputs=mute_button)
        pause_button.click(toggle_pause, outputs=pause_button)
        voice_model.change(update_voice_model, inputs=[voice_model], outputs=[gr.Textbox(label="Model Update Status")])
        
        def start_story(ref_audio, style, voice_model, scenario, custom_scenario, enable_image_generation):
            global_state.story_service = StoryService(global_state.llm_selection, global_state.groq_api_key)  # Create StoryService here
            
            for user_input, story_output, audio_output, indicator, image_output in interactive_storyteller_func(ref_audio, style, voice_model, scenario, custom_scenario, enable_image_generation):
                yield user_input, story_output, audio_output, indicator, image_output

        start_button.click(start_story, 
                           inputs=[ref_audio, style, voice_model, scenario, custom_scenario, enable_image_generation], 
                           outputs=[user_input, story_output, audio_output, indicator, image_output])
    
    demo.queue()
    demo.launch(debug=False, share=False)


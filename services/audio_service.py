# --- services/audio_service.py ---
import os
import torch
import sounddevice as sd
import soundfile as sf
import logging
import queue
import time
import io
import requests
import tempfile
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from RealtimeSTT import AudioToTextRecorder
from TTS.api import TTS
import gc
from memory_profiler import profile
from config import global_state
import asyncio
import edge_tts

class XTTSEngine:
    @profile
    def __init__(self, model_path, speaker_wav_path, language="en"):
        logging.info('Initializing TTS engine')
        self.tts = TTS(model_path, gpu=torch.cuda.is_available())
        self.speaker_wav = speaker_wav_path
        self.language = language
        self.sample_rate = 24000  # XTTS default sample rate
        self.channels = 1  # Explicitly set to mono

    @profile
    def generate_audio(self, text):
        logging.info(f'Generating audio for text: {text}')
        output = io.BytesIO()
        self.tts.tts_to_file(text=text,
                             file_path=output,
                             speaker_wav=self.speaker_wav,
                             language=self.language)
        output.seek(0)
        audio, _ = sf.read(output)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Convert to mono if stereo
        return audio

class AudioService:
    @profile
    def __init__(self, en_ckpt_base, ckpt_converter, output_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.en_ckpt_base = en_ckpt_base
        self.ckpt_converter = ckpt_converter
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_model = "OpenVoice"  # Default model
        self.xttsv2_engine = None
        self.yourtts_engine = None
        self.openvoice_model = None
        self.tone_color_converter = None
        self.temp_files = []

    @profile
    def set_model(self, model_name):
        if model_name in ["OpenVoice", "xttsv2", "yourtts", "UnrealSpeech", "edge_tts"]:
            self.current_model = model_name
            if self.current_model == "OpenVoice":
                if self.openvoice_model is None:
                    self.openvoice_model = BaseSpeakerTTS(f'{self.en_ckpt_base}/config.json', device=self.device)
                    self.openvoice_model.load_ckpt(f'{self.en_ckpt_base}/checkpoint.pth')
                    self.tone_color_converter = ToneColorConverter(f'{self.ckpt_converter}/config.json', device=self.device)
                    self.tone_color_converter.load_ckpt(f'{self.ckpt_converter}/checkpoint.pth')
                    self.en_source_se = torch.load(f'{self.en_ckpt_base}/en_default_se.pth').to(self.device)
            elif self.current_model == "xttsv2":
                if self.xttsv2_engine is None:
                    # Use ref_audio only if it's a valid file path
                    if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                        self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=ref_audio)
                    else:
                        # If ref_audio is not a valid WAV file, don't use it
                        self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path="")
            elif self.current_model == "yourtts":
                if self.yourtts_engine is None:
                    # Use ref_audio only if it's a valid file path
                    if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                        self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path=ref_audio)
                    else:
                        # If ref_audio is not a valid WAV file, don't use it
                        self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path="")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @profile
    def generate_and_play_audio(self, text, ref_audio, style):
        logging.info(f'Generating audio for text: {text}')
        save_path = os.path.join(self.output_dir, 'output.wav')  # Correctly construct path

        if self.current_model == "OpenVoice":
            if self.openvoice_model is not None:
                src_path = os.path.join(self.output_dir, 'tmp.wav')  # Correctly construct path
                self.openvoice_model.tts(text, src_path, speaker=style, language='en')
                target_se, _ = se_extractor.get_se(ref_audio, self.tone_color_converter, target_dir='processed', vad=True)
                self.tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=self.en_source_se,
                    tgt_se=target_se,
                    output_path=save_path,
                    message="@MyShell"
                )
        elif self.current_model == "xttsv2":
            if self.xttsv2_engine is None:
                # Use ref_audio only if it's a valid file path
                if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                    self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=ref_audio)
                else:
                    # If ref_audio is not a valid WAV file, don't use it
                    self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path="")
            audio = self.xttsv2_engine.generate_audio(text)
            sf.write(save_path, audio, self.xttsv2_engine.sample_rate)
        elif self.current_model == "yourtts":
            if self.yourtts_engine is None:
                # Use ref_audio only if it's a valid file path
                if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                    self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path=ref_audio)
                else:
                    # If ref_audio is not a valid WAV file, don't use it
                    self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path="")
            audio = self.yourtts_engine.generate_audio(text)
            sf.write(save_path, audio, self.yourtts_engine.sample_rate)
        elif self.current_model == "edge_tts":  # New case for edge_tts
            voice = "en-US-SteffanNeural"  # Set the desired voice
            output_file = os.path.join(self.output_dir, 'output.wav')  # Save as mp3 file

            async def generate_audio():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_file)

            # Run the async function
            asyncio.run(generate_audio())
            save_path = output_file  # Update save path to the edge_tts output

        elif self.current_model == "UnrealSpeech":
            url = "https://api.v7.unrealspeech.com/speech"
            payload = {
                "Text": text,
                "VoiceId": "Dan",  # You can make this configurable
                "Bitrate": "192k",
                "Speed": "0",
                "Pitch": "1",
                "TimestampType": "sentence"
            }
            
            self.headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {global_state.unrealspeech_api_key}"  # Access API key from GlobalState
            }
            response = requests.post(url, json=payload, headers=self.headers)
            if response.status_code == 200:
                audio_url = response.json().get("OutputUri")
                if audio_url:
                    audio_response = requests.get(audio_url)
                    if audio_response.status_code == 200:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                            temp_file.write(audio_response.content)
                            save_path = temp_file.name
                            self.temp_files.append(save_path)
                    else:
                        logging.error("Failed to download audio from UnrealSpeech")
                        return None
                else:
                    logging.error("No OutputUri in UnrealSpeech response")
                    return None
            else:
                logging.error(f"Error from UnrealSpeech API: {response.text}")
                return None

        # Play the generated audio
        try:
            data, samplerate = sf.read(save_path)
            sd.play(data, samplerate)
            sd.wait()
        except sf.LibsndfileError as e:
            print(f"Error playing audio: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        return save_path

    @profile
    def stop_audio(self):
        sd.stop()

    @profile
    def set_volume(self, volume):
        sd.default.volume = volume

    @profile
    def cleanup(self):
        """Clean up resources and temporary files."""
        logging.info("Cleaning up resources...")
        
        # Stop any playing audio
        self.stop_audio()

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Delete temporary files
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
                logging.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logging.error(f"Error deleting temporary file {temp_file}: {e}")

        self.temp_files.clear()

        # Clear other large objects
        if self.xttsv2_engine:
            del self.xttsv2_engine
            self.xttsv2_engine = None
        if self.yourtts_engine:
            del self.yourtts_engine
            self.yourtts_engine = None
        if self.openvoice_model:
            del self.openvoice_model
            self.openvoice_model = None
        if self.tone_color_converter:
            del self.tone_color_converter
            self.tone_color_converter = None

        # Manually trigger garbage collection

        gc.collect()

        logging.info("Cleanup completed")


class EnhancedAudioToTextRecorder(AudioToTextRecorder):
    @profile
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reset_time = time.time()
        self.reset_interval = 40  # Reset every 40 seconds if no input
        self.max_record_time = 30  # Maximum recording time in seconds

    @profile
    def reset(self):
        self.is_recording = False
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.wakeword_detected = False
        self.wake_word_detect_time = 0
        self.frames.clear()
        self.audio_buffer.clear()
        self._set_state("inactive")
        self.last_reset_time = time.time()
    
    @profile
    def clear_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    @profile
    def check_and_reset(self):
        current_time = time.time()
        if current_time - self.last_reset_time > self.reset_interval:
            logging.info("No input detected for a while, resetting recorder")
            self.reset()
            return True
        return False

    @profile
    def text(self):
        start_time = time.time()
        while time.time() - start_time < self.max_record_time:
            text = super().text()
            if text.strip():
                return text.strip()
            time.sleep(0.1)
        logging.info("Max recording time reached, stopping recording")
        self.stop()
        return super().text().strip()



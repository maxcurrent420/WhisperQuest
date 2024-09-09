import os
import tempfile
import torch
import sounddevice as sd
import soundfile as sf
import logging
import queue
import time
import nltk
import io
import requests
import tempfile
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter
from RealtimeSTT import AudioToTextRecorder
from TTS.api import TTS
import gc
from config import global_state
import edge_tts
import asyncio
from melo.api import TTS as MeloTTS
from memory_profiler import profile

class XTTSEngine:
    @profile
    def __init__(self, model_path, speaker_wav_path, language="en"):
        logging.info('Initializing TTS engine')
        self.tts = TTS(model_path, gpu=torch.cuda.is_available())
        self.speaker_wav_path = speaker_wav_path  # Store the speaker_wav path
        self.language = language
        self.sample_rate = 24000  # XTTS default sample rate
        self.channels = 1  # Explicitly set to mono
        # Add playback speed attribute
        self.playback_speed = 1.0

    @profile
    def generate_audio(self, text, speaker_wav=None):
        """Generates audio with optional speaker cloning.
        
        Args:
            text: The text to synthesize.
            speaker_wav: Optional path to a WAV file for voice cloning.
        
        Returns:
            The generated audio data.
        """
        logging.info(f'Generating audio for text: {text}')
        output = io.BytesIO()
        
        # If speaker_wav is provided and a valid path, use it for cloning
        if speaker_wav and os.path.exists(speaker_wav):
            self.tts.tts_to_file(text=text,
                                 file_path=output,
                                 speaker_wav=speaker_wav,  # Use the provided speaker_wav
                                 language=self.language)
        else:
            # If no speaker_wav is provided, use the default speaker
            self.tts.tts_to_file(text=text,
                                 file_path=output,
                                 language=self.language)

        output.seek(0)
        audio, _ = sf.read(output)
        if audio.ndim > 1:
            audio = audio[:, 0]  # Convert to mono if stereo
        return audio

class MeloTTSEngine:
    @profile
    def __init__(self, language='EN', device='cpu'):
        # Download required NLTK resources
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
        try:
            nltk.data.find('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('tokenizers/punkt/english.pickle')
        except LookupError:
            nltk.download('punkt_tab')

        self.model = MeloTTS(language=language, device=device)
        self.speaker_ids = self.model.hps.data.spk2id
        self.speed = 1.0

    @profile
    def generate_audio(self, text, accent='EN-US', output_path='output.wav'):
        self.model.tts_to_file(text, self.speaker_ids[accent], output_path, speed=self.speed)
        return output_path
        
    @profile
    def cleanup(self):
        """
        Explicitly cleans up resources held by the MeloTTS engine.
        """
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Clear the internal model reference
                del self.model
                # Manually trigger garbage collection
                gc.collect()
                logging.info("MeloTTS engine references cleared.")
            except Exception as e:
                logging.error(f"Error clearing MeloTTS engine references: {e}")

class AudioService:
    @profile
    def __init__(self, en_ckpt_base, ckpt_converter, en_ckpt_base_v2, ckpt_converter_v2, output_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.en_ckpt_base = en_ckpt_base
        self.ckpt_converter = ckpt_converter
        self.en_ckpt_base_v2 = en_ckpt_base_v2
        self.ckpt_converter_v2 = ckpt_converter_v2
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_model = "edge_tts"  # Default model
        self.xttsv2_engine = None
        self.yourtts_engine = None
        self.openvoice_model = None
        self.tone_color_converter = None
        self.melotts_engine = None
        self.temp_files = []
        self.default_source_se = None

    @profile
    def initialize_melotts_engine(self):
        """Initializes the MeloTTS engine."""
        self.melotts_engine = MeloTTSEngine(device=self.device)

    @profile
    def cleanup_melotts_engine(self):
        """Cleans up the MeloTTS engine."""
        if self.melotts_engine:
            self.melotts_engine.cleanup()
            del self.melotts_engine
            self.melotts_engine = None

    @profile
    def cleanup_previous_model(self):
        """Cleans up the resources for the currently loaded model."""
        if self.current_model == "OpenVoiceV2":
            self.cleanup_melotts_engine()
        elif self.current_model == "xttsv2":
            if self.xttsv2_engine:
                del self.xttsv2_engine
                self.xttsv2_engine = None
        elif self.current_model == "yourtts":
            if self.yourtts_engine:
                del self.yourtts_engine
                self.yourtts_engine = None
        elif self.current_model == "MeloTTS":
            self.cleanup_melotts_engine()

    @profile
    def initialize_openvoicev2(self):
        self.tone_color_converter = ToneColorConverter(f'{self.ckpt_converter_v2}/config.json', device=self.device)
        self.tone_color_converter.load_ckpt(f'{self.ckpt_converter_v2}/checkpoint.pth')
        # Initialize the MeloTTS engine here (lazy loading)
        self.melotts_engine = MeloTTSEngine(device=self.device)
        self.default_source_se = torch.load(f'{self.en_ckpt_base_v2}/base_speakers/ses/en-us.pth', map_location=self.device)

    @profile
    def set_model(self, model_name):
        if model_name in ["OpenVoiceV2", "xttsv2", "yourtts", "UnrealSpeech", "edge_tts", "MeloTTS"]:
            # Cleanup the previous model before initializing a new one
            self.cleanup_previous_model()
            self.current_model = model_name
            if self.current_model == "OpenVoiceV2":
                # Initialize the MeloTTS engine only once
                if self.melotts_engine is None:
                    self.initialize_openvoicev2()
            elif self.current_model == "xttsv2":
                if self.xttsv2_engine is None:
                    self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path="")
            elif self.current_model == "yourtts":
                if self.yourtts_engine is None:
                    self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path="")
            elif self.current_model == "MeloTTS":
                if self.melotts_engine is None:
                    self.melotts_engine = MeloTTSEngine(device=self.device)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    @profile
    def generate_and_play_audio(self, text, ref_audio, style):
        logging.info(f'Generating audio for text: {text}')
        save_path = os.path.join(self.output_dir, 'output.wav')

        if self.current_model == "OpenVoiceV2":
            # Split the text into sentences
            try:
                sentences = nltk.sent_tokenize(text)
            except LookupError:
                nltk.download('punkt_tab')
                sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                src_path = os.path.join(self.output_dir, 'tmp.wav')
                self.melotts_engine.generate_audio(sentence, accent='EN-US', output_path=src_path)

                # Handle UploadedFile
                if hasattr(ref_audio, 'name') and hasattr(ref_audio, 'getvalue'):  # Check if it's likely an UploadedFile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_file.write(ref_audio.getvalue())
                        ref_audio_path = temp_file.name
                else:
                    ref_audio_path = ref_audio  # Assume it's already a path

                target_se, _ = se_extractor.get_se(ref_audio_path, self.tone_color_converter, target_dir='processed', vad=True)

                self.tone_color_converter.convert(
                    audio_src_path=src_path,
                    src_se=self.default_source_se,
                    tgt_se=target_se,
                    output_path=save_path,
                    message="@MyShell"
                )

                # Clean up temporary file if created
                if ref_audio_path != ref_audio:
                    os.unlink(ref_audio_path)

                # Play the generated audio for each sentence
                try:
                    data, samplerate = sf.read(save_path)
                    sd.play(data, samplerate)
                    sd.wait()
                    # Release memory immediately after playback
                    del data
                    del src_path
                    del ref_audio_path
                    del target_se
                except sf.LibsndfileError as e:
                    print(f"Error playing audio: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")

                # Ensure memory is released after each sentence
                gc.collect()
                torch.cuda.empty_cache()  # Free up GPU memory

        elif self.current_model == "xttsv2":
            if self.xttsv2_engine is None:
                self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=ref_audio)
            audio = self.xttsv2_engine.generate_audio(text)
            sf.write(save_path, audio, self.xttsv2_engine.sample_rate)
        elif self.current_model == "yourtts":
            if self.yourtts_engine is None:
                if hasattr(ref_audio, 'name') and hasattr(ref_audio, 'getvalue'):  # Check if it's an UploadedFile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                        temp_file.write(ref_audio.getvalue())
                        ref_audio_path = temp_file.name
                        self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path=ref_audio_path)
                        audio = self.yourtts_engine.generate_audio(text, speaker_wav=ref_audio_path)  # Pass the ref_audio as speaker_wav
                        sf.write(save_path, audio, self.yourtts_engine.sample_rate)
                        self.temp_files.append(temp_file.name)
                else:
                    # If ref_audio is not a valid WAV file, don't use it
                    self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path="")
            
            # Generate audio using the yourtts_engine
            if hasattr(ref_audio, 'name') and hasattr(ref_audio, 'getvalue'):  # Check if it's an UploadedFile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(ref_audio.getvalue())
                    ref_audio_path = temp_file.name
                    audio = self.yourtts_engine.generate_audio(text, speaker_wav=ref_audio_path)  # Pass the ref_audio as speaker_wav
                    sf.write(save_path, audio, self.yourtts_engine.sample_rate)
                    self.temp_files.append(temp_file.name)
            else:
                audio = self.yourtts_engine.generate_audio(text, speaker_wav=ref_audio)
                sf.write(save_path, audio, self.yourtts_engine.sample_rate)

        elif self.current_model == "edge_tts":
            voice = "en-US-EmmaNeural"
            output_file = os.path.join(self.output_dir, 'output.wav')

            async def generate_audio():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(output_file)

            asyncio.run(generate_audio())
            save_path = output_file
        elif self.current_model == "UnrealSpeech":
            url = "https://api.v7.unrealspeech.com/speech"
            payload = {
                "Text": text,
                "VoiceId": "Dan",
                "Bitrate": "192k",
                "Speed": "0",
                "Pitch": "1",
                "TimestampType": "sentence"
            }

            self.headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {global_state.unrealspeech_api_key}"
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
        elif self.current_model == "MeloTTS":
            if self.melotts_engine is None:
                self.melotts_engine = MeloTTSEngine(device=self.device)
            save_path = self.melotts_engine.generate_audio(text, accent='EN-US', output_path=save_path)

        # Play the generated audio
        try:
            data, samplerate = sf.read(save_path)
            # Apply playback speed to the samplerate
            sd.play(data, int(samplerate * global_state.playback_speed))
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
    def cleanup(self):
        logging.info("Cleaning up resources...")

        self.stop_audio()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)
                logging.info(f"Deleted temporary file: {temp_file}")
            except Exception as e:
                logging.error(f"Error deleting temporary file {temp_file}: {e}")

        self.temp_files.clear()

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
        # Cleanup the MeloTTS engine before exiting
        self.cleanup_melotts_engine()
        
        gc.collect()

        logging.info("Cleanup completed")

class EnhancedAudioToTextRecorder(AudioToTextRecorder):
    @profile
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reset_time = time.time()
        self.reset_interval = 50
        self.max_record_time = 45

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

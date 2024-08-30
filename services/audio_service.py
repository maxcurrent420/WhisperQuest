# --- services/audio_service.py ---
import os
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

class XTTSEngine:
    def __init__(self, model_path, speaker_wav_path, language="en"):
        logging.info('Initializing TTS engine')
        self.tts = TTS(model_path, gpu=torch.cuda.is_available())
        self.speaker_wav = speaker_wav_path
        self.language = language
        self.sample_rate = 24000  # XTTS default sample rate
        self.channels = 1  # Explicitly set to mono

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

class MeloTTSEngine:
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
        
        self.model = MeloTTS(language=language, device=device)
        self.speaker_ids = self.model.hps.data.spk2id
        self.speed = 1.0

    def generate_audio(self, text, accent='EN-US', output_path='output.wav'):
        self.model.tts_to_file(text, self.speaker_ids[accent], output_path, speed=self.speed)
        return output_path

class AudioService:
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
        self.melotts_engine = None
        self.temp_files = []

    def set_model(self, model_name):
        if model_name in ["OpenVoice", "xttsv2", "yourtts", "UnrealSpeech", "edge_tts", "MeloTTS"]:
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
                    if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                        self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=ref_audio)
                    else:
                        self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path="")
            elif self.current_model == "yourtts":
                if self.yourtts_engine is None:
                    if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                        self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path=ref_audio)
                    else:
                        self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path="")
            elif self.current_model == "MeloTTS":
                if self.melotts_engine is None:
                    self.melotts_engine = MeloTTSEngine()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_and_play_audio(self, text, ref_audio, style):
        logging.info(f'Generating audio for text: {text}')
        save_path = os.path.join(self.output_dir, 'output.wav')

        if self.current_model == "OpenVoice":
            if self.openvoice_model is not None:
                src_path = os.path.join(self.output_dir, 'tmp.wav')
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
                if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                    self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path=ref_audio)
                else:
                    self.xttsv2_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav_path="")
            audio = self.xttsv2_engine.generate_audio(text)
            sf.write(save_path, audio, self.xttsv2_engine.sample_rate)
        elif self.current_model == "yourtts":
            if self.yourtts_engine is None:
                if os.path.exists(ref_audio) and ref_audio.lower().endswith(".wav"): 
                    self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path=ref_audio)
                else:
                    self.yourtts_engine = XTTSEngine(model_path="tts_models/multilingual/multi-dataset/your_tts", speaker_wav_path="")
            audio = self.yourtts_engine.generate_audio(text)
            sf.write(save_path, audio, self.yourtts_engine.sample_rate)
        elif self.current_model == "edge_tts":
            voice = "en-US-SteffanNeural"
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
                self.melotts_engine = MeloTTSEngine()
            save_path = self.melotts_engine.generate_audio(text, accent='EN-US', output_path=save_path)

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

    def stop_audio(self):
        sd.stop()

    def set_volume(self, volume):
        sd.default.volume = volume

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
        if self.melotts_engine:
            del self.melotts_engine
            self.melotts_engine = None

        gc.collect()

        logging.info("Cleanup completed")

class EnhancedAudioToTextRecorder(AudioToTextRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reset_time = time.time()
        self.reset_interval = 40
        self.max_record_time = 30

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
    
    def clear_audio_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def check_and_reset(self):
        current_time = time.time()
        if current_time - self.last_reset_time > self.reset_interval:
            logging.info("No input detected for a while, resetting recorder")
            self.reset()
            return True
        return False

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

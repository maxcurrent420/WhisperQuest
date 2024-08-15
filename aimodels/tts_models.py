import os
import torch
import io
import soundfile as sf
from TTS.api import TTS
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

class TTSModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def generate_audio(self, text, ref_audio, style):
        raise NotImplementedError("Subclasses must implement this method")
print(f"Current voice model: {global_state.selected_voice_model}")
class OpenVoiceModel(TTSModel):
    def __init__(self, en_ckpt_base, ckpt_converter, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        self.en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
        self.en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
        self.tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        self.tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        self.en_source_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
print(f"Current voice model: {global_state.selected_voice_model}")
    def generate_audio(self, text, ref_audio, style):
        src_path = f'{self.output_dir}/tmp.wav'
        save_path = f'{self.output_dir}/output.wav'

        self.en_base_speaker_tts.tts(text, src_path, speaker=style, language='en')

        target_se, _ = se_extractor.get_se(ref_audio, self.tone_color_converter, target_dir='processed', vad=True)

        self.tone_color_converter.convert(
            audio_src_path=src_path,
            src_se=self.en_source_se,
            tgt_se=target_se,
            output_path=save_path,
            message="@MyShell"
        )

        return save_path

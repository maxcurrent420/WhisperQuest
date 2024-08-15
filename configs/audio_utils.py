import queue
import logging
import time
from RealtimeSTT import AudioToTextRecorder

class EnhancedAudioToTextRecorder(AudioToTextRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reset_time = time.time()
        self.reset_interval = 40  # Reset every 40 seconds if no input
        self.max_record_time = 30  # Maximum recording time in seconds

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

def create_audio_recorder():
    return EnhancedAudioToTextRecorder(
        model="tiny",
        language="en",
        spinner=True,
        wake_word_activation_delay=5
        
        

    )

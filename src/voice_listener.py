import pyaudio
import json
import vosk
import numpy as np

class VoiceListener:
    def __init__(self, model_path="vosk-model-en-us-0.22"):
        print(f"Loading Vosk model from '{model_path}'...")
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        print("Vosk model loaded.")

        self.p_audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_size = 4096

    def listen_and_transcribe(self, listening_prompt="Listening for command..."):
        stream = self.p_audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        print(listening_prompt)
        print("Speak now...")

        transcript = ""
        while not transcript:
            data = stream.read(self.chunk_size, exception_on_overflow=False)

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                recognized_text = result.get('text', '').strip()
                if recognized_text:
                    transcript = recognized_text
        
        stream.stop_stream()
        stream.close()
        
        return transcript

    def cleanup(self):
        self.p_audio.terminate()
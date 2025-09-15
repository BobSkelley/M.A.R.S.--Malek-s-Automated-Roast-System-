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
        self.chunk_size = 4096  # Increased chunk size for better accuracy
        self.silence_timeout = 1.5  # Seconds of silence to stop listening

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
        
        frames = []
        silence_count = 0
        max_silence = int(self.silence_timeout * self.sample_rate / self.chunk_size)
        
        while True:
            data = stream.read(self.chunk_size)
            frames.append(data)
            
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                if 'text' in result and result['text'].strip():
                    transcript = result['text'].strip()
                    break
            
            # Check for silence to timeout
            audio_data = np.frombuffer(data, dtype=np.int16)
            if np.abs(audio_data).mean() < 500:  # Silence threshold
                silence_count += 1
            else:
                silence_count = 0
                
            if silence_count > max_silence:
                transcript = ""
                break
        
        stream.stop_stream()
        stream.close()
        
        # Final result check if no result was accepted during streaming
        if 'transcript' not in locals():
            final_result = json.loads(self.recognizer.FinalResult())
            transcript = final_result.get('text', '').strip()
        
        return transcript

    def cleanup(self):
        self.p_audio.terminate()
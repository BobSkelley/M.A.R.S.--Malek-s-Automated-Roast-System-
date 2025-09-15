import os
import time
from llama_cpp import Llama
from TTS.api import TTS
from playsound import playsound
import torch
import tempfile

class SarcasticAI:
    def __init__(self, model_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        print("Loading LLaMA model (Mistral 7B)...")
        start_time = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=35,
            verbose=False
        )
        print(f"LLaMA model loaded in {time.time() - start_time:.2f}s")

        print("Loading Coqui XTTS model...")
        start_time = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print(f"Coqui TTS model loaded in {time.time() - start_time:.2f}s")

        self.system_prompt = "You are MARS, Malek's Automated Roast System. Your personality is witty, sarcastic, and satirical. Your goal is to affectionately roast the user with short, clever insults. Never be a generic AI assistant. Keep your responses concise."
        self.history = []

    def generate_dynamic_line(self, context="boot"):
        if context == "boot":
            prompt = f"[INST] {self.system_prompt}\n\nGenerate a sarcastic, witty boot-up line for MARS. Keep it under 2 sentences. [/INST]\nMARS:"
        else:
            prompt = f"[INST] {self.system_prompt}\n\nGenerate a sarcastic, witty shutdown line for MARS. Keep it under 2 sentences. [/INST]\nMARS:"
        
        output = self.llm(
            prompt,
            max_tokens=100,
            stop=["</s>", "[INST]"],
            temperature=0.8,
            echo=False
        )
        return output['choices'][0]['text'].strip()

    def generate_response(self, user_input, speech_patterns=None):
        start_time = time.time()
        
        history_string = ""
        for message in self.history[-4:]:
            role = "User" if message["role"] == "user" else "MARS"
            history_string += f"{role}: {message['content']}\n"
        
        pattern_prompt = ""
        if speech_patterns:
            pattern_prompt = f"\nNote: The user tends to use these words/phrases: {speech_patterns}. Incorporate some of these naturally into your response to mock their speech patterns."
        
        prompt = f"[INST] {self.system_prompt}{pattern_prompt}\n\n{history_string}User: {user_input} [/INST]\nMARS:"
        
        output = self.llm(
            prompt,
            max_tokens=150,
            stop=["</s>", "[INST]", "User:"],
            temperature=0.7,
            echo=False
        )
        
        response_text = output['choices'][0]['text'].strip()
            
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response_text})
        self.history = self.history[-6:]

        print(f"LLM response generated in {time.time() - start_time:.2f}s")
        return response_text

    def speak(self, text_to_speak):
        start_time = time.time()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_file_path = temp_audio_file.name
            
            self.tts.tts_to_file(
                text=text_to_speak,
                file_path=temp_file_path,
                speaker="Ana Florence",
                language="en"
            )
            
            tts_time = time.time() - start_time
            print(f"TTS generated in {tts_time:.2f}s")
            
            # Add delay before playing to ensure file is fully written
            time.sleep(0.5)
            playsound(temp_file_path, block=True)
            
            for _ in range(5):
                try:
                    os.remove(temp_file_path)
                    break
                except PermissionError:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

    def cleanup(self):
        pass
from sarcastic_ai import SarcasticAI
from voice_listener import VoiceListener
from vocabulary_manager import VocabularyManager
import time
import os

def main():
    # Define the base directory of the project
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define paths to models and data
    llama_model_path = os.path.join(base_dir, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    vosk_model_path = os.path.join(base_dir, "models", "vosk-model-en-us-0.22")
    db_path = os.path.join(base_dir, "data", "vocabulary.db")

    start_time = time.time()
    ai = SarcasticAI(model_path=llama_model_path)
    listener = VoiceListener(model_path=vosk_model_path)
    vocab_manager = VocabularyManager(db_path=db_path)
    print(f"System initialized in {time.time() - start_time:.2f}s")

    print("\n--- M.A.R.S. Initialized ---")

    print("Generating boot line...")
    boot_start = time.time()
    boot_line = ai.generate_dynamic_line("boot")
    print(f"AI: {boot_line}")
    ai.speak(boot_line)
    print(f"Boot line generated in {time.time() - boot_start:.2f}s")

    try:
        while True:
            listen_start = time.time()
            user_speech = listener.listen_and_transcribe()
            listen_time = time.time() - listen_start

            if user_speech:
                print(f"\nYou said: '{user_speech}' (transcribed in {listen_time:.2f}s)")

                vocab_start = time.time()
                vocab_manager.log_phrase(user_speech)
                print(f"Vocabulary logged in {time.time() - vocab_start:.2f}s")

                if "quit" in user_speech.lower() or "stop" in user_speech.lower():
                    print("Generating shutdown line...")
                    shutdown_start = time.time()
                    shutdown_line = ai.generate_dynamic_line("shutdown")
                    print(f"AI: {shutdown_line}")
                    ai.speak(shutdown_line)
                    print(f"Shutdown line generated in {time.time() - shutdown_start:.2f}s")
                    break
                else:
                    print("AI is thinking...")
                    response_start = time.time()

                    user_patterns = vocab_manager.get_user_speech_patterns()
                    print(f"User patterns detected: {user_patterns}")

                    text_response = ai.generate_response(user_speech, user_patterns)
                    response_time = time.time() - response_start

                    print(f"AI: {text_response} (total response time: {response_time:.2f}s)")

                    speak_start = time.time()
                    ai.speak(text_response)
                    speak_time = time.time() - speak_start
                    print(f"Speech synthesized in {speak_time:.2f}s")
            else:
                print("No speech detected. Please try again.")

    except KeyboardInterrupt:
        print("\nForce shutdown initiated.")
    finally:
        listener.cleanup()
        ai.cleanup()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()
# main_window.py
import tkinter as tk
import os
import random
import glob
from input.audio_capture import AudioCapture
from asr.whisper_asr import WhisperASR
from api_parser.openai_parser import OpenAIParser
from nlp.named_entity_recognition import named_entity_recognizer
from nlp.scene_understanding import scene_processor
from image_generator.text_to_image import TextToImage
from database.db_manager import DatabaseManager
from gui.display_image import Display
from utils.utils import scene_to_text, numerical_sort

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Adventure Art")

        # Initialize other components
        self.db_manager = DatabaseManager()
        self.scene_processor_instance = scene_processor(self.db_manager)
        self.named_entity_recognizer_instance = named_entity_recognizer(self.db_manager)
        self.parser_instance = OpenAIParser()
        self.image_generator = TextToImage()
        self.display = Display(root)

        self.debug = True  # Set debug mode
        self.file_pointer = 0  # Initialize file pointer
        self.labeled_audio_files = sorted(glob.glob('dnd_sample/*.wav'), key=numerical_sort)
        self.current_scene = None

        if self.debug:        
            self.file_pointer = random.randint(0, len(self.labeled_audio_files) - 1)  # Randomly initialize the file pointer
            
        # Setup GUI components
        self.setup_gui()


    def setup_gui(self):
        # Create and place GUI components here
        # Example: Button to start processing
        self.start_button = tk.Button(self.root, text="Start", command=self.start_processing)
        self.start_button.pack()

    def start_processing(self):
        self.periodic_update()

    def periodic_update(self):
        # Check if all files have been processed in debug mode
        if self.debug and self.file_pointer >= len(self.labeled_audio_files):
            return 

        self.process_audio_file()
        self.root.after(1000, self.periodic_update)  # Schedule the next update

    def process_audio_file(self):
        # 1. Capture audio
        audio_file_path = self.capture_audio()

        # 2. Transcribe audio to text
        transcription = self.transcribe_audio(audio_file_path)

        # 3. Scene Understanding
        self.scene_processor_instance.process_scene_text(transcription)

        # 4. Named Entity Recognition
        self.named_entity_recognizer_instance.identify_named_entities(transcription)

        # 5. Update current scene
        self.current_scene = self.db_manager.get_current_scene()

        # 6. Generate a descriptive prompt for image generation
        prompt = self.generate_prompt(self.current_scene, transcription)

        if prompt:
            self.generate_and_display_image(prompt)

        # Clean up the temporary audio file
        self.cleanup_audio_file(audio_file_path)

    def capture_audio(self):
        if self.debug:
            audio_file_path = self.labeled_audio_files[self.file_pointer]
            self.file_pointer = (self.file_pointer + 1) % len(self.labeled_audio_files)
        else:
            capturer = AudioCapture(debug=True)
            audio_file_path = capturer.capture_audio(30)
            capturer.close()
        print(f"Audio captured and saved to {audio_file_path}")
        return audio_file_path

    def transcribe_audio(self, audio_file_path):
        asr = WhisperASR("medium.en")
        transcription = asr.transcribe(audio_file_path)
        print(f"Transcription: {transcription}")
        return transcription

    def generate_prompt(self, current_scene, transcription):
        prompt = self.parser_instance.generate_prompt(current_scene, transcription)
        if not prompt:
            print("Text is not visualizable")
        else:
            print(f"Generated Prompt: {prompt}")
        return prompt

    def generate_and_display_image(self, prompt):
        image_path = self.image_generator.generate_image(prompt)
        print(f"Image generated and saved to {image_path}")
        self.display.show_image(image_path)

    def cleanup_audio_file(self, audio_file_path):
        if not self.debug:
            os.remove(audio_file_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()

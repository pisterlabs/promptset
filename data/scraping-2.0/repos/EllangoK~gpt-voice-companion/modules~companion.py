import json
import logging
import os
import time
from enum import Enum

import gradio as gr
import speech_recognition as sr
from colorama import Fore, Style, init
from playaudio import playaudio

from modules.config_manager import ConfigManager
from modules.elevenlabs import ElevenLabsTTS
from modules.history_manager import HistoryEntry, HistoryManager
from modules.openai_wrapper import OpenAI
from modules.utils import ensure_dir_exists

init(autoreset=True)  # Initialize colorama


class Companion:

    AUDIO_PATH = "audio/"
    CONVERSATION_PATH = "conversations/"

    DEFAULTS = {
        'voice_recognition': True,
        'gui': False
    }

    def __init__(self, args_dict: dict, debug: bool = True):
        self.config_manager = ConfigManager(args_dict, self.DEFAULTS)
        self.load_config()
        self.openai = OpenAI(args_dict)
        self.elevenlabs = ElevenLabsTTS(args_dict)
        self.history_manager = HistoryManager()

        ensure_dir_exists(self.AUDIO_PATH)
        ensure_dir_exists(self.CONVERSATION_PATH)
    
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.CRITICAL)

    def load_config(self):
        self.voice_recognition = self.config_manager['voice_recognition']
        self.gui = self.config_manager['gui']
        self.save_config()

    def save_config(self):
        self.config_manager['voice_recognition'] = self.voice_recognition
        self.config_manager['gui'] = self.gui

    def get_response(self, prompt: str) -> str:
        response = self.openai.query_gpt(prompt)
        logging.debug(f"Response: {response}")
        return response

    def say(self, text: str):
        success, audio_path = self.elevenlabs.write_audio(text, self.AUDIO_PATH)
        self.history_manager.add_audio_path_to_last_entry(audio_path, False)
        logging.debug(f"Audio path: {audio_path}")
        if success:
            print(Fore.BLUE + f"{self.openai.name}: " + Style.RESET_ALL + text)
            playaudio(audio_path)
        else:
            print(Fore.RED + "Error: " + Style.RESET_ALL + "Could not access ElevenLabs API.")

    def process_input(self, text: str, is_voice: bool = False, retry_attempts: int = 3):
        if is_voice:
            if text.lower() == "exit":
                return ProcessStatus(ProcessInputStatus.EXIT, None)
            elif text.lower() == "help":
                return ProcessStatus(ProcessInputStatus.LOG, "Commands for voice input:\nexit - quit\nhelp - help")
        else:
            if text.startswith("!"):
                if text[1].lower() == "q":
                    return ProcessStatus(ProcessInputStatus.EXIT, None)
                elif text[1].lower() == "h":
                    return ProcessStatus(ProcessInputStatus.LOG, "Commands for text input:\n!q - quit\n!h - help")

        for attempt in range(retry_attempts + 1):
            response = self.get_response(self.history_manager.to_str() + text)
            if len(response.strip()) > 0:
                break
            if attempt == retry_attempts:
                return ProcessStatus(ProcessInputStatus.LOG, (Fore.RED + "Error: " + Style.RESET_ALL + "No response from chatbot."))
            logging.debug(f"Retry attempt {attempt + 1}")

        user_entry = HistoryEntry(f"User: {text}", None)
        ai_entry = HistoryEntry(f"{self.openai.name}: {response}", None)
        self.history_manager.add_entry(user_entry, ai_entry)
        logging.debug(f"History: {self.history_manager.to_str()}")

        return ProcessStatus(ProcessInputStatus.SAY, response)

    def loop_text_input(self):
        while True:
            user_input = input(Fore.GREEN + "You: " + Style.RESET_ALL)
            logging.debug(f"User input: {user_input}")

            status = self.process_input(user_input)
            if status.status == ProcessInputStatus.EXIT:
                break
            elif status.status == ProcessInputStatus.LOG:
                print(status.response)
            elif status.status == ProcessInputStatus.SAY:
                self.say(status.response)

    def loop_voice_input(self):
        recognizer = sr.Recognizer()
        while True:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source)
                try:
                    user_input = recognizer.recognize_google(audio)
                    logging.debug(f"Audio user input: {user_input}")
                    print(Fore.GREEN + "You: " + Style.RESET_ALL + user_input)
                except sr.UnknownValueError:
                    print(Fore.RED + "Error: " + Style.RESET_ALL + "Could not understand the audio.")
                    continue
                except sr.RequestError:
                    print(Fore.RED + "Error: " + Style.RESET_ALL + "Could not request results from Google Speech Recognition service.")
                    continue

            status = self.process_input(user_input, is_voice=True)
            if status.status == ProcessInputStatus.EXIT:
                break
            elif status.status == ProcessInputStatus.LOG:
                print(status.response)
            elif status.status == ProcessInputStatus.SAY:
                self.say(status.response)

    def chatbot_tab(self):
        with gr.Tab("Chatbot"):
            gr.Markdown(f"""## You are talking to a chatbot named {self.openai.name}, prompted with:
            \"{self.openai.context}\"""")
            chatbot = gr.Chatbot(label=self.openai.name)
            text_input = gr.Textbox(show_label=False)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
                exit_btn = gr.Button("Exit")

            def user(user_message, chatbot_dialogue):
                return "", chatbot_dialogue + [[user_message, None]]

            def bot(chatbot_dialogue):
                status = self.process_input(chatbot_dialogue[-1][0])
                if status.status == ProcessInputStatus.SAY:
                    self.say(status.response)
                chatbot_dialogue[-1][1] = status.response
                return chatbot_dialogue
            
            def clear_func():
                self.history_manager.clear()
                return None

            global exit_check
            exit_check = False

            def exit_func():
                global exit_check
                exit_check = True
                print(Fore.RED + "Exiting..." + Style.RESET_ALL)
                return None

            text_input.submit(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            submit_btn.click(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(
                bot, chatbot, chatbot
            )
            clear_btn.click(fn=clear_func, inputs=None, outputs=chatbot, queue=False)
            exit_btn.click(fn=exit_func, inputs=None, outputs=None, queue=False)

    def conversation_explorer_tab(self):
        with gr.Tab("Conversation Explorer"):
            pass

    def launch_demo(self):
        global exit_check

        with gr.Blocks(title="Voice Assistant") as demo:
            self.chatbot_tab()
            self.conversation_explorer_tab()

        demo.launch(prevent_thread_lock=True)

        while not exit_check:
            time.sleep(0.5)

        demo.close()

    def loop(self):
        if self.gui:
            self.launch_demo()
        else:
            print("Type !h for help")
            print(Fore.YELLOW + f"Context: " + Style.RESET_ALL + "You are talking to a chatbot named " + Fore.BLUE + self.openai.name + Style.RESET_ALL + f", prompted with \"{self.openai.context}\".")

            if self.voice_recognition:
                self.loop_voice_input()
            else:
                self.loop_text_input()

    def get_voices(self):
        return self.elevenlabs.get_voices()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.save_conversation()

    def save_conversation(self):
        date = time.strftime("%Y-%m-%d", time.localtime())
        folder_path = os.path.join(self.CONVERSATION_PATH, date)
        ensure_dir_exists(folder_path)
        path = os.path.join(folder_path, f"{str(int(time.time()))}.json")
 
        with open(path, "w", encoding="utf8") as f:
            f.write(self.history_manager.to_json())

class ProcessInputStatus(Enum):
    SAY = 0
    LOG = 1
    EXIT = 2

class ProcessStatus:

    def __init__(self, status: ProcessInputStatus, data: str = None):
        self.status = status
        self.response = data

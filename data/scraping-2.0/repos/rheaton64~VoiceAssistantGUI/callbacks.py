import queue
import threading
import datetime
import json
import os
import subprocess
import time
from pywinauto import Application
from elevenlabs import generate, stream, set_api_key, voices
from langchain.callbacks.base import BaseCallbackHandler
from action.action_executor_new import ActionExecutor

audio_queue = queue.Queue()

# Generates an audio stream from text and adds it to the audio queue
def enqueue_generation(text: str, voice: str, display_queue: queue.Queue):
    display_queue.put(text)
    audio = generate(
        text=text,
        voice=voice,
        stream=True,
    )
    audio_queue.put(audio)

# Plays audio from the audio queue
def play_audio_from_queue(playing_audio):
    while True:
        audio = audio_queue.get()
        playing_audio.set()
        stream(audio)
        if audio_queue.qsize() == 0:
            playing_audio.clear()
        audio_queue.task_done()

class AssistantCallbackHandler(BaseCallbackHandler):

    def __init__(self, voice, api_key, running_event, playing_event, action_pending, action_queue, display_queue):
        set_api_key(api_key)
        voices()
        threading.Thread(target=play_audio_from_queue, daemon=True, args=[playing_event]).start()
        self.running_event = running_event
        self.playing_event = playing_event
        self.log_info = {
            'prompts': [],
            'response': None,
            'actions': [],
            'utterances': [],
        }
        self.output_handler = AssistantOutputHandler(voice, self.log_info, self.playing_event, action_pending, action_queue, display_queue)

    def save_log(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"response_log_{current_time}.json"
        file_path = os.path.join('./response_logs', filename)
        
        with open(file_path, 'w+') as gen_file:
            gen_file.write(json.dumps(self.log_info, indent=4))

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.running_event.set()
        self.log_info['prompts'] = prompts
        self.output_handler.start()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.log_info['utterances'].append(token)
        self.output_handler.send_token(token)

    def on_llm_end(self, response, **kwargs) -> None:
        self.log_info['response'] = str(response)
        self.output_handler.flush()
        self.running_event.clear()
        self.save_log()

class AssistantOutputHandler():
    def __init__(self, voice, log_info, is_playing, action_pending, action_queue, display_queue):
        self.voice = voice
        self.token_buffer = []
        self.sentence_buffer = ""
        self.is_code = False
        self.is_action = False
        self.action_buffer = "<ACTION>"
        self.is_playing = is_playing
        self.log_info = log_info
        self.action_pending = action_pending
        self.display_queue = display_queue
        self.action_executor = ActionExecutor(action_pending=action_pending, action_queue=action_queue)

    def start(self):
        pass

    def send_action(self, action):
        action = action.replace('<ACTION>', '')
        action = action.replace('</ACTION>', '')
        threading.Thread(target=self.action_executor.send, args=[action]).start()

        self.log_info['actions'].append(action)

    def check_action(self, token):
        is_action_start = False
        is_action_end = False
        temp_buffer = self.token_buffer.copy()
        temp_buffer.append(token)
        joined_temp_buffer = ''.join(temp_buffer)

        if '<ACTION>' in joined_temp_buffer:
            self.is_action = True
            is_action_start = True
            self.token_buffer.clear()
            return True
        elif '</ACTION>' in self.action_buffer:
            is_action_end = True
            self.token_buffer.clear()

        if self.is_action:
            if is_action_start:
                self.action_buffer = '<ACTION>'
                is_action_start = False
            else:
                self.action_buffer += token
            if is_action_end:
                self.is_action = False
                is_action_end = False
                self.send_action(self.action_buffer)
                self.action_buffer = ''
            return True
        
        return False

    def check_code(self, token):
        is_code_end = False
        if token in ['```', '`\n\n']: # Check if token is a code block delimiter

            if not self.is_code:  # If it's the start of a code snippet
                self.is_code = True
                if self.token_buffer:  # If there's text in the buffer
                    self.sentence_buffer += ''.join(self.token_buffer) # Add it to the sentence buffer
                    self.token_buffer.clear()
                self.sentence_buffer += " I'm writing the code to the window now." # Add a message to the sentence buffer
                enqueue_generation(self.sentence_buffer, self.voice, self.display_queue) # Generate and play the sentence buffer
                self.sentence_buffer = "" # Clear the sentence buffer

            else:  # If it's the end of a code snippet
                is_code_end = True

        if self.is_code:  # If the LLM is currently outputting a code snippet

            if is_code_end:  # If it's the end of the code snippet
                self.is_code = False
                is_code_end = False
            return True
        return False
    
    def check_markdown(self, token):
        if '**' in token:
            return True
        return False
    
    def send_token(self, token):

        if self.check_code(token):
            return
        
        if self.check_action(token):
            return
        
        if self.check_markdown(token):
            return
        
        self.token_buffer.append(token)
        if token.endswith(('.', '?', '!', '"', '\n', ':')):
            self.sentence_buffer += ''.join(self.token_buffer)
            self.token_buffer.clear()
        if self.sentence_buffer and audio_queue.qsize() == 0 and not self.is_playing.is_set():
            enqueue_generation(self.sentence_buffer, self.voice, self.display_queue)
            self.sentence_buffer = ""

    def flush(self):
        if self.sentence_buffer:
            enqueue_generation(self.sentence_buffer, self.voice, self.display_queue)
            self.sentence_buffer = ""
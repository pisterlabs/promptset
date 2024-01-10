import queue
import subprocess
import threading

import openai

from voice_bot_demo.console import console
from voice_bot_demo.text_speech.interface import TextSpeechInterface
from voice_bot_demo.utils import file_relative_path


class GPT_PYTTSX3TextSpeech(TextSpeechInterface):
    def __init__(self):
        self._messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self._text_queue = queue.Queue()

        text_processing_thread = threading.Thread(target=self._process_text_queue)
        text_processing_thread.start()

    def _chat_gpt_request(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
        assistant_reply = response.choices[0].message["content"]
        return assistant_reply

    def _process_text_queue(self):
        while True:
            text = self._text_queue.get()
            console.print(
                f"[bold green]User[/bold green]: {text}",
                highlight=False,
                soft_wrap=True,
            )

            if text is None:
                break

            self._messages.append({"role": "user", "content": text})
            gpt_response = self._chat_gpt_request(self._messages)
            console.print(
                f"[bold red]Assistant[/bold red]: {gpt_response}",
                highlight=False,
                soft_wrap=True,
            )

            self._messages.append({"role": "assistant", "content": gpt_response})
            subprocess.run(
                ["python", file_relative_path(__file__, "tts.py"), gpt_response]
            )

    def process_sentence(self, text):
        self._text_queue.put(text)

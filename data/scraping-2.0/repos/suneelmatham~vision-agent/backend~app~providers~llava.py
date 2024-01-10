from app.api_models import ChatGPTConversationItem, ChatGPTConversationsModel
from typing import Literal
from app.prompts import system_prompt as SYSTEM_PROMPT
import json
import requests as requests
from openai import OpenAI
import os
import base64
from app.utils import TimeStats


class ConversationAdapter():
    def __init__(self, conversation_data: ChatGPTConversationsModel):
        self.conversation_data = conversation_data
        self.stop_sequence = "###"
        self.completion_delimiter = "$$$"

    def generate_prompt(self):
        system_prompt = ""
        conversation_prompt = ""
        for message in self.conversation_data.data[0:-1]:
            if message.role == "system":
                system_prompt += message.content
            if message.role == "user":
                conversation_prompt += "Human:" + message.content + self.stop_sequence
            if message.role == "assistant":
                conversation_prompt += "Assistant:" + message.content + self.stop_sequence
        if not system_prompt:
            system_prompt += f"""{SYSTEM_PROMPT} {self.stop_sequence}Human: Who are you?{self.stop_sequence}Assistant: I am helpful assistant who will help you in any task{self.stop_sequence}Human: What is your name{self.stop_sequence}Assistant: No, you wouldn't get it
            """
        conversation_prompt = conversation_prompt.replace("<image>", "")
        image_count = len(self.conversation_data.images)
        last_message = self.conversation_data.data[-1]
        question = "Human:" + ("<image>" * image_count) + last_message.content
        return system_prompt + self.stop_sequence + conversation_prompt + question + self.completion_delimiter + f"ASSISTANT:"

    def extract_answer(self, text: str):
        answer = text.split(self.completion_delimiter)[-1]
        return answer.replace("ASSISTANT:", "")

    def update_conversation(self, text: str):
        answer = text.split(self.completion_delimiter)[-1]
        self.add_message('assistant', answer)
        return answer

    def add_message(self, role: Literal['user', 'assistant'], text: str):
        self.conversation_data.data.append(
            ChatGPTConversationItem(role=role, content=text))


openai_client = OpenAI()


def text_to_audio(text_chunk: str):
    audio_stream = openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text_chunk
    )
    return base64.b64encode(audio_stream.read()).decode('utf-8')


sentence_delimiters = (". ", ".", ",", "?", "!", ";",
                       ":", "—", "-", "(", ")", "[", "]", "}")


def generate_llava_response(query: ChatGPTConversationsModel):
    conversation_adapter = ConversationAdapter(query)
    images = [image.replace("data:image/png;base64,", "")
              for image in query.images]
    worker_url = f"{os.getenv('BAKLLAVA_MODEL_WORKER_ADDR')}/worker_generate_stream"
    prompt = conversation_adapter.generate_prompt()
    request_payload = {
        "model": "llava-v1.5-13b",
        "prompt": prompt,
        "max_new_tokens": 100,
        "temperature": 0.3,
        "stop": conversation_adapter.stop_sequence,
        "images": images
    }
    request_headers = {'Content-Type': 'application/json'}
    time_stats = TimeStats()
    response = requests.post(
        worker_url, headers=request_headers, json=request_payload, stream=True)
    response_buffer = b""
    parsed_response = b""
    response_offset = 0
    response_text = ""
    for chunk in response.iter_content():
        time_stats.end_language()
        response_buffer += chunk
        if chunk == b"\x00":
            parsed_response = response_buffer.replace(b"\x00", b"")
            response_buffer = b""
            try:
                response_extract = json.loads(parsed_response.decode())
                response_text = conversation_adapter.extract_answer(
                    response_extract['text']).strip()
                if response_text.endswith(sentence_delimiters) and len(response_text[response_offset::].replace(" ", "")) > 0:
                    audio_base64 = text_to_audio(
                        response_text[response_offset::])
                    time_stats.end_speech()
                    response_data = {
                        "text": response_text,
                        "audio": audio_base64,
                        "ttfb_speech": time_stats.get_speech_ttfb(),
                        "ttfb_language": time_stats.get_language_ttfb()
                    }
                    response_offset = len(response_text)
                    yield json.dumps([response_data]) + "\x00"
            except:
                pass
    if response_text[response_offset::] or len(response_text[response_offset::]) > 0:
        response_data = {
            "text": response_text,
            "audio": text_to_audio(response_text[response_offset::]),
            "ttfb_speech": time_stats.get_speech_ttfb(),
            "ttfb_language": time_stats.get_language_ttfb()
        }
        yield json.dumps([response_data]) + "\x00"

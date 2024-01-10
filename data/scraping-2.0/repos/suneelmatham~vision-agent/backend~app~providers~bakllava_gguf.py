from app.api_models import ChatGPTConversationItem, ChatGPTConversationsModel
from typing import Literal
from app.prompts import system_prompt as SYSTEM_PROMPT
import json
import requests as r
import pprint
import re
import os
import base64
from openai import OpenAI
from app.utils import TimeStats


URL = f'{os.getenv("BAKLLAVA_GGUF_MODEL_WORKER_ADDR")}/completion'


# This class is just a adapter class so that we can convert the incoming conversations into a prompt which llava can use
# incoming data will be in format of JSON/Object (same as how we send to OpenAI API)


class BakllavaConversations():

    def __init__(self, conversations: ChatGPTConversationsModel):
        self.conversations = conversations
        self.delimiterForCompletion = "$$$"
        self.stop = "###"

    def get_prompt(self):
        system_prompt = ""
        ret = ""
        # print(self.conversations.data)
        image_number = 1
        for message in self.conversations.data[0:-1]:
            if message.role == "system":
                system_prompt += message.content
            if message.role == "user":
                ret += "Human:" + message.content + self.stop
            if message.role == "assistant":
                ret += "Assistant:" + message.content + self.stop

            image_number += 1

        if not system_prompt:
            system_prompt += f"""{SYSTEM_PROMPT} {self.stop}Human: Who are you?{self.stop}Assistant: I am a helpful assistant who can help you in any task{self.stop}Human: What is your name{self.stop}Assistant: No, you wouldn't get it
            """
        ret = re.sub(r'\[img-\d+\]', '', ret)

        images_placeholder = ""
        for i in range(len(self.conversations.images)):
            images_placeholder += f"[img-{i + 1}]\n"

        last_message = self.conversations.data[-1]
        question = "Human:" + images_placeholder + last_message.content

        return system_prompt + self.stop + ret + question + f"###ASSISTANT:"

    def update_conversations_from_text_stream(self, text: str):
        answer = text.split(self.stop)[-1]
        self.add_message('assistant', answer)
        return answer

    def add_message(self, role: Literal['user', 'assistant'], text: str):
        self.conversations.data.append(
            ChatGPTConversationItem(role=role, content=text))
        # print(messages)

    def getJson(self):
        return self.conversations.json()


def get_images(query: ChatGPTConversationsModel):
    images = []
    image_id = 1
    for image in query.images:
        images.append({"data": image.replace(
            "data:image/png;base64,", ""), "id": image_id})
        image_id += 1

    return images


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


def stream_bakllava_text_response(query: ChatGPTConversationsModel):
    conversations = BakllavaConversations(query)
    images = get_images(query)

    prompt = conversations.get_prompt()
    payload = {
        "prompt": prompt,
        "n_predict": 128,
        "image_data": images,
        "stream": True
    }
    headers = {
        'Content-Type': 'application/json'
    }
    time_stats = TimeStats()

    response = r.post(URL, headers=headers, json=payload, stream=True)
    answer = ""
    offset = 0
    for chunk in response.iter_content(chunk_size=128):
        time_stats.end_language()
        content = chunk.decode().strip().split('\n\n')[0]
        try:
            content_split = content.split('data: ')
            if len(content_split) > 1:
                content_json = json.loads(content_split[1])
                answer = answer + content_json["content"]
                print(answer)
                if (str(answer).strip().endswith(sentence_delimiters)) and len(answer[offset::].replace(" ", "")) > 0:
                    audio_base64 = text_to_audio(
                        answer[offset::])
                    time_stats.end_speech()
                    response_data = {
                        "text": answer,
                        "audio": audio_base64,
                        "ttfb_speech": time_stats.get_speech_ttfb(),
                        "ttfb_language": time_stats.get_language_ttfb()
                    }
                    yield json.dumps([response_data]) + "\x00"

        except json.JSONDecodeError:
            break
    print("Number of images sent", len(images))

    conversations.add_message("assistant", answer)

    if answer[offset::] or len(answer[offset::]) > 0:
        response = {
            "text": answer,
            "audio": text_to_audio(answer[offset::]),
            "ttfb_speech": time_stats.get_speech_ttfb(),
            "ttfb_language": time_stats.get_language_ttfb()
        }

        yield json.dumps([response])
    # yield json.dumps([""])

    # convert the text back to speech and send back to client

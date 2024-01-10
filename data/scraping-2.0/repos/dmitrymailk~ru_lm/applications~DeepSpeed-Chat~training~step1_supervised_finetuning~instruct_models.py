import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import gc
from transformers import StoppingCriteria, StoppingCriteriaList

import requests
import json
import uuid
import time


import openai


class GoralConversation:
    def __init__(
        self,
        message_template=" <s> {role}\n{content} </s>\n",
        system_prompt="Ты — Горал, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
        start_token_id=1,
        bot_token_id=9225,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode(
            [
                self.start_token_id,
            ]
        )
        final_text += " "
        final_text += tokenizer.decode([self.bot_token_id])
        return final_text.strip()


class SaigaConversation:
    def __init__(
        self,
        message_template="<s> {role}\n{content}</s>\n",
        system_prompt="Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.",
        start_token_id=1,
        bot_token_id=9225,
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]) :]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


def add_special_tokens_v2(string):
    string = string.replace("\n", "</s>")
    return string


def remove_special_tokens_v2(string):
    string = string.replace("</s>", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace("<|endoftext|>", "")
    return string


def encode_v2(text: str, tokenizer, special_tokens=True):
    text = add_special_tokens_v2(text)
    text = tokenizer.encode(text, add_special_tokens=special_tokens)
    return text


def decode_v2(tokens: list[int], tokenizer):
    tokens = tokenizer.decode(tokens)
    tokens = remove_special_tokens_v2(tokens)
    return tokens


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops, tokenizer, prompt):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
        self.prompt = add_special_tokens_v2(prompt)
        self.prompt = tokenizer.decode(tokenizer.encode(self.prompt))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            generated_temp_ids = input_ids.tolist()[0]
            if stop in self.tokenizer.decode(generated_temp_ids)[len(self.prompt) :]:
                return True

        return False


class XGLMConversation:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        debug_status: int = 0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.debug_status = debug_status
        self.max_history = 3

        self.history = []

    def chat(
        self,
        user_message: str,
    ) -> str:
        self.history.append(
            {
                "source": "user",
                "message": user_message,
            },
        )
        total_prompt = ""
        self.history = self.history[-2 * self.max_history :]

        if self.debug_status:
            print(self.history)

        for item in self.history:
            message = item["message"]
            if item["source"] == "user":
                total_prompt += f"\nHuman:\n{message}"
            else:
                total_prompt += f"\nAssistant:\n{message}"

        total_prompt += "\nAssistant:\n"
        if self.debug_status:
            print(total_prompt)
            print("=" * 100)

        answer = self.generate_response(total_prompt)
        answer = self.extract_answer(
            answer,
            prev_prompt=total_prompt,
        )
        self.history.append(
            {
                "source": "bot",
                "message": answer,
            },
        )
        return answer

    def generate_response(self, prompt):
        stop_words = [
            "<|endoftext|>",
            "Human:",
        ]
        stopping_criteria = StoppingCriteriaList(
            [
                StoppingCriteriaSub(
                    stops=stop_words,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                )
            ]
        )
        gen_config = GenerationConfig(
            max_new_tokens=2048,
            repetition_penalty=1.1,
            eos_token_id=[400],
        )

        with torch.no_grad():
            input_text = encode_v2(
                prompt,
                tokenizer=self.tokenizer,
            )
            input_text = torch.tensor([input_text]).to("cuda")

            output_tokens = self.model.generate(
                input_text,
                generation_config=gen_config,
                stopping_criteria=stopping_criteria,
            )
            finetuned_result = decode_v2(output_tokens[0], tokenizer=self.tokenizer)
            torch.cuda.empty_cache()
            gc.collect()
            return finetuned_result

    def start_chat(self):
        while True:
            message = input("You: ")

            if self.debug_status == 1:
                print(message)
                print("-" * 100)

            if message == "exit":
                break
            answer = self.chat(message)

            if self.debug_status:
                print("CONTEXT:", self.history)

            if self.last_response == answer:
                self.history = []
            else:
                self.last_response = answer

            print("Bot:", answer)

    def extract_answer(self, g_answer: str, prev_prompt: str = None):
        answer = g_answer[len(prev_prompt) :].strip()
        answer = answer.replace("Human:", " ")
        answer = answer.replace("Assistant:", " ")
        return answer


class GigaChatConversationAPI:
    def __init__(
        self,
        creds_path: str = "./gigachad.json",
        session_id=None,
    ) -> None:
        self.creds_path = creds_path
        self.headers = json.loads(open(self.creds_path).read())
        self.current_conversation = None

        # для того чтобы начать новую беседу нужно пересоздать
        # объект класса
        if session_id is None:
            session_id = str(uuid.uuid1())
        self.current_conversation = session_id

    def send_message(self, user_message: str):
        self.headers[
            "Referer"
        ] = f"https://developers.sber.ru/studio/workspaces/69a30c72-a3ac-43c6-8c5e-821383a4f064/ml/projects/70bc01c6-9062-4aa6-9c77-26c550fb4440/sessions/{self.current_conversation}"
        payload = {
            "generate_alternatives": False,
            "preset": "default",
            "request_json": user_message,
            "session_id": self.current_conversation,
            "model_type": "GigaChat:v1.13.0",
        }
        # print("PAYLOAD: ", payload)
        response = requests.post(
            "https://developers.sber.ru/api/chatwm/api/client/request",
            headers=self.headers,
            data=json.dumps(payload),
        )
        response = response.json()
        # print("RESPONSE: ", response)

        request_id = self.get_request_id()
        self.request_id = request_id
        bot_response = self._get_bot_response(request_id=request_id)
        return bot_response

    def get_request_id(
        self,
    ):
        payload = {
            "offset": 0,
            "limit": 26,
            "session_id": "a63627a1-7261-40a5-ae0b-00bafaadc542",
            "newer_first": True,
        }
        payload["session_id"] = self.current_conversation
        self.headers[
            "Referer"
        ] = f"https://developers.sber.ru/studio/workspaces/69a30c72-a3ac-43c6-8c5e-821383a4f064/ml/projects/70bc01c6-9062-4aa6-9c77-26c550fb4440/sessions/{self.current_conversation}"
        # print(payload)

        response = requests.post(
            "https://developers.sber.ru/api/chatwm/api/client/session_messages",
            headers=self.headers,
            data=json.dumps(payload),
        )
        request_id = response.json()["messages"][0]["request_id"]

        return request_id

    def _get_bot_response(self, request_id: str):
        space_id = self.headers["Space-Id"]
        user_id = self.headers["User-Id"]
        request_str = f"https://developers.sber.ru/api/chatwm/api/client/get_result_events?request_id={request_id}&space-id={space_id}&user-id={user_id}"
        # print(request_str)
        response = requests.get(
            request_str,
            headers=self.headers,
        )
        self.bot_response = response
        response = list(response.iter_lines())
        # print(response)
        response = response[-2]
        response = response.decode("utf-8")[5:].strip()
        response = json.loads(response)
        response = response["responses"][0]["data"]
        # print("BOT RESPONSE", response)
        return response


class YandexGPTAPI:
    def __init__(
        self,
        creds_path="./yandexGPT.json",
        generation_config_path="./yandexGPT_generation_params.json",
    ) -> None:
        self.creds_path = creds_path
        self.generation_config_path = generation_config_path

        self.headers = json.loads(open(self.creds_path).read())
        self.generation_config = json.loads(open(self.generation_config_path).read())

        self.chat_history = []

    def send_instruct(
        self,
        message: str,
    ):
        data = {
            "model": "general",
            "generationOptions": self.generation_config,
            "instructionText": message,
        }

        result = requests.post(
            "https://llm.api.cloud.yandex.net/llm/v1alpha/instruct",
            headers=self.headers,
            data=json.dumps(data),
        )
        text = result.json()
        text = text["result"]["alternatives"][0]["text"]
        return text

    def send_chat(self, message: str):
        time.sleep(1)
        new_message = {
            "role": "user",
            "text": message,
        }
        self.chat_history.append(new_message)

        data = {
            "model": "general",
            "generationOptions": self.generation_config,
            "messages": self.chat_history,
        }

        result = requests.post(
            "https://llm.api.cloud.yandex.net/llm/v1alpha/chat",
            headers=self.headers,
            data=json.dumps(data),
        )
        text = result.json()
        if not "result" in text:
            print(text)
        new_message = text["result"]["message"]
        text = text["result"]["message"]["text"]
        self.chat_history.append(new_message)
        return text

    def clear_chat(self):
        self.chat_history = []


class ChatGPTConversationAPI:
    def __init__(
        self,
        api_key_path="./chat_gpt_token",
    ):
        self.api_key = open(api_key_path).read()
        openai.api_key = self.api_key
        self.chat_history = []

    def send_message(self, message):
        self.chat_history.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.chat_history
        )
        self.chat_history.append(
            {"role": "assistant", "content": response.choices[0].message.content}
        )
        return response.choices[0].message.content

    def clear_chat_history(self):
        self.chat_history = []

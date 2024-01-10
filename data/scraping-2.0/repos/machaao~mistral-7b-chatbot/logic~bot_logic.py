import base64
import json
import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from requests.structures import CaseInsensitiveDict
from transformers import AutoTokenizer

ban_words = ["nigger", "negro", "nazi", "faggot", "murder", "suicide"]

# list of banned input words
c = 'UTF-8'


def send(url, headers, payload=None):
    if payload:
        print("sending post to platform: " + str(payload))
        response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
        print("response from the platform: " + str(response.text))
    else:
        response = requests.request("GET", url, headers=headers)

    return response


# don't change for sanity purposes
def get_details(api_token, base_url):
    _cache_ts_param = str(datetime.now().timestamp())
    e = "L3YxL2JvdHMvY29uZmlnP3Y9"
    check = base64.b64decode(e).decode(c)
    url = f"{base_url}{check}{_cache_ts_param}"
    headers = {
        "api_token": api_token,
        "Content-Type": "application/json",
    }

    response = send(url, headers)

    if response and response.status_code == 200:
        return response.json()
    else:
        return {}


class BotLogic:
    def __init__(self, server_session_create_time):
        # Initializing Config Variables
        load_dotenv()

        self.api_token = os.environ.get("API_TOKEN")
        self.base_url = os.environ.get("BASE_URL", "https://ganglia.machaao.com")
        self.name = os.environ.get("NAME")
        self.limit = os.environ.get("LIMIT", 'True')
        self.server_session_create_time = server_session_create_time
        self.model = os.environ.get("MODEL_NAME")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        # Bot config
        self.top_p = os.environ.get("TOP_P", 1.0)
        self.top_k = os.environ.get("TOP_K", 20)
        self.temp = os.environ.get("TEMPERATURE", 0.3)
        self.max_length = os.environ.get("MAX_LENGTH", 50)
        self.validate_bot_params()

    # noinspection DuplicatedCode
    def validate_bot_params(self):
        print("Setting up Bot server with parameters:")
        if self.top_p is not None and self.temp is not None:
            print("Temperature and Top_p parameters can't be used together. Using default value of top_p")
            self.top_p = 0.5

        if self.temp is not None:
            self.temp = float(self.temp)
            if self.temp < 0.0 or self.temp > 1.0:
                raise Exception("Temperature parameter must be between 0 and 1")
        else:
            self.temp = 0.8
        print(f"Temperature = {self.temp}")

        if self.top_p is not None:
            self.top_p = float(self.top_p)
            if self.top_p <= 0.0 or self.top_p >= 1.0:
                raise Exception("Top_p parameter must be between 0 and 1")
        else:
            self.top_p = 1.0
        print(f"Top_p = {self.top_p}")

        if self.top_k is not None:
            self.top_k = int(self.top_k)
            if self.top_k > 1000:
                raise Exception("Top_k parameter must be less than 1000")
        else:
            self.top_k = 50
        print(f"Top_k = {self.top_k}")

        if self.max_length is not None:
            self.max_length = int(self.max_length)
            if self.max_length > 1024:
                raise Exception("Max_length parameter must be less than 1024")
        else:
            self.max_length = 50
        print(f"Max_Length = {self.max_length}")

    @staticmethod
    def read_prompt(name):
        file_name = "./logic/prompt.txt"
        with open(file_name) as f:
            prompt = f.read()

        return prompt.replace("[name]", f"{name}")

    def get_recent(self, user_id: str, current_session=True):
        limit = 5
        url = f"{self.base_url}/v1/conversations/history/{user_id}/{limit}"

        headers = CaseInsensitiveDict()
        headers["api_token"] = self.api_token
        headers["Content-Type"] = "application/json"

        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            messages = resp.json()
            if current_session:
                filtered_messages = list()
                for message in messages:
                    create_time_stamp = message.get("_created_at")
                    create_time = datetime.strptime(create_time_stamp, "%Y-%m-%dT%H:%M:%S.%fZ")
                    if create_time > self.server_session_create_time:
                        filtered_messages.append(message)

                while len(filtered_messages) > 0 and (filtered_messages[0].get("type") == "outgoing"):
                    _ = filtered_messages.pop(0)

                return filtered_messages
            else:
                return messages

    @staticmethod
    def parse(data):
        msg_type = data.get('type')
        if msg_type == "outgoing":
            msg_data = json.loads(data['message'])
            msg_data_2 = json.loads(msg_data['message']['data']['message'])

            if msg_data_2 and msg_data_2.get('text', ''):
                text_data = msg_data_2['text']
            elif msg_data_2 and msg_data_2['attachment'] and msg_data_2['attachment'].get('payload', '') and \
                    msg_data_2['attachment']['payload'].get('text', ''):
                text_data = msg_data_2['attachment']['payload']['text']
            else:
                text_data = ""
        else:
            msg_data = json.loads(data['incoming'])
            if msg_data['message_data']['text']:
                text_data = msg_data['message_data']['text']
            else:
                text_data = ""

        return msg_type, text_data

    def core(self, req: str, label: str, user_id: str, client: str, sdk: str, action_type: str, api_token: str):
        print(
            "input text: " + req + ", label: " + label + ", user_id: " + user_id + ", client: " + client + ", sdk: " + sdk
            + ", action_type: " + action_type + ", api_token: " + api_token)

        bot = get_details(api_token, self.base_url)
        name = self.name

        if not bot:
            return False, "Oops, the chat bot doesn't exist or is not active at the moment"
        else:
            name = bot.get("displayName", name)

        valid = True

        recent_text_data = self.get_recent(user_id)
        recent_convo_length = len(recent_text_data)

        print(f"len of history: {recent_convo_length}")

        banned = any(ele in req for ele in ban_words)

        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": f"assistant", "content": "I'm doing great. How can I help you today?"}
        ]

        if banned:
            print(f"banned input:" + str(req) + ", id: " + user_id)
            return False, "Oops, please refrain from such words"

        for text in recent_text_data[::-1]:
            msg_type, text_data = self.parse(text)

            if text_data:
                e_message = "Oops," in text_data and "connect@machaao.com" in text_data

                if msg_type is not None and not e_message:
                    # outgoing msg - bot msg
                    messages.append({
                        "role": f"assistant",
                        "content": text_data
                    })
                else:
                    # incoming msg - user msg
                    messages.append({
                        "role": "user",
                        "content": text_data
                    })

        try:
            reply = self.process_via_huggingface(name, messages)
            return valid, reply
        except Exception as e:
            print(f"error - {e}, for {user_id}")
            return False, "Oops, I am feeling a little overwhelmed with messages\nPlease message me later"

    def process_via_huggingface(self, name, messages):
        parameters = {
            "max_new_tokens": self.max_length,
            "temperature": self.temp,
            "top_k": self.top_k,
            "use_cache": False,
            "top_p": self.top_p
        }

        print(f"processing for {self.model}")

        hub_llm = HuggingFaceHub(repo_id=self.model, model_kwargs=parameters)


        templ = self.tokenizer.apply_chat_template(messages, tokenize=False)

        _prompt = self.read_prompt(name)

        prompt = PromptTemplate(
            input_variables=["conversation"],
            template=_prompt + ": {conversation}"
        )

        hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
        resp = hub_chain.run(templ)
        return resp

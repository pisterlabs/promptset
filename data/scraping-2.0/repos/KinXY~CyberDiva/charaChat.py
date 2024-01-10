import string
from chat import Chat
from prompts import (
    get_begin_prompts,
    get_tone_prompts,
    filter_sayings,
    filter_info_points,
    combine_sayings,
)
from read import get_chara_setting_keys
import openai
from openai.embeddings_utils import get_embedding
import os
import json
import asyncio
import websockets
import time


class CharaChat(Chat):
    def __init__(self, charaSet: dict, chatSet: dict, userSet: dict):
        super().__init__(chatSet)
        self.chara = charaSet
        self.user = userSet
        self.real_history = []
        self.filtered_setting = []

    def get_filtered_setting(self, input: string):
        TOTAL_LENGTH = 5000
        self.filtered_setting = {}
        # get the keys in the charaInit for character setting
        keys = get_chara_setting_keys(self.chara["name"])
        # allocate the number of sayings to be filtered for each key
        values_total_length = 0
        for key in keys:
            for value in self.chara[key]:
                values_total_length += len(value["content"])
        # filter the settings
        for key in keys:
            values = self.chara[key]
            filter_num = int(
                TOTAL_LENGTH
                * (len(values) / values_total_length)
            )
            filtered_values = filter_sayings(
                sayings=values, 
                input=input,
                api_key=self.setting["api_key"],
                num=filter_num,
            )
            self.filtered_setting[key] = combine_sayings(filtered_values)

    def user_input(self, input: string):
        start_time = time.time()
        self.get_filtered_setting(input)

        init_msg = get_begin_prompts(
            charaSet=self.chara,
            userSet=self.user,
            filtered_setting=self.filtered_setting,
        )
        named_input = self.user["name"] + ": " + input

        if self.history == []:
            for msg in init_msg:
                self.history.append(msg)
        else:
            for i, msg in enumerate(init_msg):
                self.history[i] = msg
        super().user_input(named_input)
        self.real_history.append(
            with_embedding({"role": "user", "content": input}, self.setting["api_key"])
        )
        print("user_input: " + str(time.time() - start_time))

    def add_response(self, response: string):
        start_time = time.time()
        response = filter_info_points(
            info_points=response,
            input=self.history[-1]["content"],
            api_key=self.setting["api_key"],
            charaSet=self.chara,
        )
        print("filter_info_points: " + str(time.time() - start_time))
        start_time = time.time()
        tone_response = openai.Completion.create(
            model="text-davinci-003",
            prompt=get_tone_prompts(
                setting=self.setting,
                charaSet=self.chara,
                userSet=self.user,
                history=self.real_history,
                info_points=response,
                filtered_setting=self.filtered_setting,
                api_key=self.setting["api_key"],
            ),
            max_tokens=self.setting["max_tokens"],
            temperature=self.setting["temperature"],
            presence_penalty=self.setting["presence_penalty"],
        )

        tone_text = tone_response["choices"][0]["text"]
        tone_text = clean_response(tone_text)

        self.history.append({"role": "assistant", "content": response})
        self.real_history.append(
            with_embedding(
                {
                    "role": "assistant",
                    "content": tone_text,
                },
                self.setting["api_key"],
            )
        )
        print("add_response: " + str(time.time() - start_time))
        return pair_response_list(
            response_list=seperate_response(tone_text, self.chara)
        )

    def print_history(self):
        # os.system("cls")
        for _msg in self.real_history:
            msg = _msg["content"]
            if msg["role"] == "user":
                print("You: " + msg["content"])
            else:
                print(self.chara["name"] + ": " + msg["content"])

    def trigger_live2d(self, response_pairs: list):
        async def send_message(myMotion, myText):
            motion_msg = {
                "msg": 13200,
                "msgId": 1,
                "data": {"id": 0, "type": 0, "mtn": myMotion},
            }
            text_msg = {
                "msg": 11000,
                "msgId": 1,
                "data": {
                    "id": 0,
                    "text": myText,
                    "textFrameColor": 0x000000,
                    "textColor": 0xFFFFFF,
                    "duration": 10000,
                },
            }
            async with websockets.connect("ws://127.0.0.1:10086/api") as ws:
                if myMotion != "":
                    await ws.send(json.dumps(motion_msg))
                if myText != "":
                    await ws.send(json.dumps(text_msg))

        loop = asyncio.get_event_loop()
        for response in response_pairs:
            if response["motion"] != "":
                motion = filter_sayings(
                    sayings=self.chara["motions"],
                    input=response["motion"],
                    api_key=self.setting["api_key"],
                    num=1,
                )[0]["content"]
            else:
                motion = ""
            loop.run_until_complete(send_message(motion, response["text"]))
            sleep_time = 0.1 * len(response["text"])
            loop.run_until_complete(asyncio.sleep(sleep_time))


# HELPER FUNCTIONS
def with_embedding(msg: dict, api_key: string):
    openai.api_key = api_key
    embedding = get_embedding(text=msg["content"], engine="text-embedding-ada-002")
    return {"content": msg, "embedding": embedding}


def clean_response(response: string):
    # delete the nonsense at the beginning of the response
    for i in range(len(response)):
        if not response[i] in ["\n", " ", '"']:
            response = response[i:]
            break
    # delete the nonsense at the end of the response
    for i in range(len(response) - 1, -1, -1):
        if not response[i] in ["\n", " ", '"']:
            response = response[: i + 1]
            break
    # clear all the \" in the response
    response = response.replace('\"', "")
    return response


def seperate_response(response: string, charaSet: dict):
    response_list = []
    # seperate the response into a list of strings by contents in brackets
    while True:
        if not "[" in response:
            if not response in ["", "\n", " ", '"']:
                response_list.append({"type": "text", "content": response})
            break
        else:
            left = response.index("[")
            right = response.index("]")
            content_in = response[left + 1 : right]
            content_before = response[:left]
            content_after = response[right + 1 :]
            if not content_before in ["", "\n", " ", '"']:
                response_list.append({"type": "text", "content": content_before})

            # remove the character name from the motion
            content_in = content_in.replace(charaSet["name"], "")
            response_list.append({"type": "motion", "content": content_in})
            response = content_after

    return response_list


def pair_response_list(response_list: list):
    response_pairs = []
    # pair the motion and text in the response_list together
    for i in range(0, len(response_list), 2):
        if i == len(response_list) - 1:
            if response_list[i]["type"] == "motion":
                motion = response_list[i]["content"]
                response_pairs.append({"motion": motion, "text": ""})
            else:
                text = response_list[i]["content"]
                response_pairs.append({"motion": "", "text": text})
            break
        if response_list[i]["type"] == "motion":
            motion = response_list[i]["content"]
            text = response_list[i + 1]["content"]
            response_pairs.append({"motion": motion, "text": text})
        else:
            motion = response_list[i + 1]["content"]
            text = response_list[i]["content"]
            response_pairs.append({"motion": motion, "text": text})
    return response_pairs

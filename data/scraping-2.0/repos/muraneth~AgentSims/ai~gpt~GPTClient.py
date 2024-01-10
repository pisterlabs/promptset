"""
Content: Client wrapper of GPT
Author : Lu Yudong
Editor : Fisher
"""

import os
import sys

abs_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
import openai
import time
from typing import Any, List, Dict, Tuple
from config import Config


class GPTClient:
    def __init__(self, cfg: Config) -> None:
        self.config = cfg
        self.keys = cfg.gpt_api_keys
        self.timer = dict()
        self.use_proxy = cfg.gpt_use_proxy
        self.proxy = cfg.gpt_proxy

    def send_and_recv(self, msg: List[Dict[str, Any]], model: str) -> Tuple[List[str], Exception]:
        if self.use_proxy:
            self.set_proxy()
        keys = self.keys[model]
        api_key = keys[0]
        # time_to_wait = 0
        # for key in keys:
        #     if model == "gpt-3.5-turbo":
        #         wait = self.config.gpt3_cooldown - (time.time() - self.timer.get(key, 0))
        #         if wait <= 0:
        #             time_to_wait = 0
        #             api_key = key
        #             break
        #         elif time_to_wait == 0 or (time_to_wait > 0 and wait <= time_to_wait):
        #             time_to_wait = wait
        #             api_key = key
        #     else:
        #         api_key = key
        #         break
        #         time.sleep(self.config.gpt3_cooldown - diff)
        # if model == "gpt-3.5-turbo":
        #     self.timer[api_key] = time.time()
        # if time_to_wait > 0:
        #     time.sleep(time_to_wait)
        openai.api_key = api_key
        # print("msg:", msg)
        with open(os.path.join(abs_path, "counter.txt"), "a", encoding="utf-8") as counter_file:
            info = ""
            for m in msg:
                info += m["content"]
            print(info)
            info = repr(info) + "\n"
            counter_file.write(info)
        counter = 0
        while counter < 3:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=msg,
                    temperature=self.config.gpt_temp,
                )
        # if model == "gpt-3.5-turbo":
        #     self.timer[api_key] = time.time()
                result = response.choices[0]["message"]["content"]
                break
            except openai.error.RateLimitError as e:
                print(e)
                counter += 1
        if self.use_proxy:
            self.unset_proxy()
        # print("result:", result)
        with open(os.path.join(abs_path, "counter.txt"), "a", encoding="utf-8") as counter_file:
            info = result
            print(info)
            info = repr(info) + "\n"
            counter_file.write(info)
        return result

    def set_proxy(self) -> None:
        os.environ["https_proxy"] = self.proxy
        os.environ["http_proxy"] = self.proxy

    def unset_proxy(self) -> None:
        os.environ["https_proxy"] = ""
        os.environ["http_proxy"] = ""


if __name__ == '__main__':
    cfg = Config(os.path.join(abs_path, "..", "config", "app.json"))
    client = GPTClient(cfg)
    # for _ in range(4):
    # print(client.send_and_recv([{'role': 'system', 'content': '\nYou are a game character who can reponse to other charcter and you will be provided with a retrived memory. \nYou should not know anything beyond your character settings or memory. \nAll your actions must conform to your personality. Your personality is measured using data from the MBTI personality test. Your personality is represented by four variables, all ranging from 1 to 10, where 1 to 10 represents a uniform transition relationship. In the first number, 1 represents Extravert and 10 represents Introvert. The second number 1 represents Sensing, and 10 represents Intuition. The third number 1 represents Intuition and 10 represents Feeling. The fourth number 1 represents the judgment, and 10 represents Perceiving.\nYour personality is :\n```{\n3,7,5,5\n}```\nBased on your personality, your daily behavior also needs to conform to the worldview of the game. Of course, different personalities are allowed to have different understandings of the worldview. The worldview of the game is as follows:\n```{\nLive a daily life.\n}```\nYou have also been provided with a biography to introduce your identity, profession, and social relationships. This will serve as your basic introduction and cannot be changed. It is:\n```{\n"Name": John Lin\n"Age": 40\n"Carrer":cafe owner\n"Social Relationship":John Lin is living with his wife, Mei Lin, who is a park manager, and son, Eddy Lin, who is a student studying music theory; John Lin loves his family very much\n"Other Introduction":He loves to help people. He is always looking for ways to make the process of getting good-taste coffee easier for his customers\n}```\n\n'}, {'role': 'user', 'content': '\nYour sight is:```\nYou got Lake,Eddy Lin,Bed,Table,Cabinet,Bookshelf,Park,House,Mei Lin nearby.current time is: 2023-05-16 16:36.Mei Lin\'s current status is: Wandering.Eddy Lin\'s current status is: Wandering.\n```\nYour most relevant memories are:```\n\n```\nYour plan is:```\nPrepare the cafe for opening\n```\nYour action is a choose to achieve your plan, which can be decomposed as a single step in the actual situation.  A single step is not necessary to achieve the whole plan.\nYour operation should be chosen from available operation list. If you choose "go to your chosen buildings", you should choose a building on the map additionally. If you choose "talk with nearby people", you should choose a person nearby from Nearby people list additionally. If you choose other operation, you shoold choose an equipment additionally.\nAvailable operation list is: ```["exercise", "go to \'your chosen buildings\'", "talk with nearby people", "have meal", "read books", "rest", "sleep"]```.\nBuildings on the map list: ```["Cafe", "House", "Park"]```.\nNearby people list is: ```["Eddy Lin", "Mei Lin"]```. \nWhat\'s your next action?Your action should be formed in JSON format, where the `location` field can be used to indicate the building to go to, \xa0`equipment` field to indicate the equipments to interact with, the `operation` field indicates the behavior to interact with, the `name` field indicates the person to communicate with, and the `content` field indicates what to say to this person. The above fields can be a null value if you do not want to perform the related operation.\n'}], "gpt-3.5-turbo"))
    # enc = tiktoken.get_encoding("cl100k_base")
    # assert enc.decode(enc.encode("hello world")) == "hello world"

    # # To get the tokeniser corresponding to a specific model in the OpenAI API:
    # print(enc = tiktoken.encoding_for_model("gpt-4"))
    result = client.send_and_recv(msg=[{"role": "system", "content": "你是一个热心的智能助手。"},
                                            {"role": "user", "content": "你好，你是谁?"}], model="gpt-3.5-turbo")
    print(result)

import json
from typing import List

import openai

from config import openai_endpoints
from openai_endpoint import OpenAIEndpoint


def get_prompt(length=1024) -> List[dict]:
    prompt_list = []
    with open("../material/prompts.json", "r") as f:
        origin_list = json.load(f)
        for data in origin_list:
            prompt_item = {}
            if len(data["origin_prompt"]) > length:
                prompt_item["prompt"] = "{}。{}".format(
                    data["origin_prompt"][:length], data["question"]
                )
            else:
                prompt_item["prompt"] = "{}。{}".format(
                    data["origin_prompt"], data["question"]
                )
            prompt_list.append(prompt_item)
    return prompt_list


class ChatCompletion:
    def __init__(self, endpoint: OpenAIEndpoint):
        self.model = endpoint.model
        self.server = endpoint.server
        self.stop = endpoint.stop

    def send_chat(self, prompt) -> str:
        openai.api_key = "sk-"
        openai.api_base = self.server
        res = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=128,
            top_p=0.8,
            # stop=self.stop,
        )

        return res["choices"][0]["message"]["content"]


if __name__ == "__main__":
    for env, endpoint in openai_endpoints.items():
        if not endpoint.enable:
            continue
        for loop in range(1, 30):
            length = 512 * loop
            if length > 9100:
                break
            prompt_list = get_prompt(length)
            for prompt_item in prompt_list:
                chat_completion = ChatCompletion(endpoint)
                print(f"env: {env}, length: {len(prompt_item['prompt'])} ,start")
                from datetime import datetime

                start_time = datetime.now()
                res = chat_completion.send_chat(prompt_item["prompt"])
                end_time = datetime.now()
                time_difference = end_time - start_time
                print(time_difference)
                print(f"env: {env}, response: {res}, time consumed: {time_difference}")

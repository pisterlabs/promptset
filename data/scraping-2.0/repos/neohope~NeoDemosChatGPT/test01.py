#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml


'''
api调用演示

# for windows
set PATH=D:/NeoLang/Python/Python310_x64;%PATH%
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
python test01.py

# for linux
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
python test01.py
'''


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


def generate_prompt(gender):
    return """Suggest three names for a Chinese boy who love cars.

Baby: Boy
Names: Tom, Jerry, Henrry
Baby: Girl
Names: Alice, Alma, Ann
Baby: {}
Names:""".format(gender)


if __name__ == '__main__':
    get_api_key()
    gender = "Boy"
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(gender),
            temperature=0.6,
        )

    print(response)


'''
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " Auton, Moton, Driven"
    }
  ],
  "created": 1679718420,
  "id": "cmpl-6xpmW7Kmmq7AepquRHEvQw62N3mkC",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 8,
    "prompt_tokens": 45,
    "total_tokens": 53
  }
}
'''


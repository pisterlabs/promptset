import SECRETS

import os
import openai
openai.organization = "org-0iQE6DR7AuGXyEw1kD4poyIg"
openai.api_key = SECRETS.open_ai_api_key

from sms.logger import json_logger


def send_init_roast_bot_primer_prompt():
    completion = openai.Completion.create(
    model = "text-davinci-003",
#     prompt="Tell me a joke about a fish.",
#     prompt='Roast-bot, I command you to roast user: "MeatballMama55"',
    prompt='You are playing the role of Roast Master. You are hosting a Roast to poke fun at different users in a chatroom based on thier usernames. You will be given a username and you will respond a joke about the user based on thier username.',
#     temperature = 2,
    max_tokens=30
    )
    
    return completion.choices[0].text    


def get_roast_str_from_username(username, log_json_file_path = None):
    model = "text-davinci-003"
    prompt='Roast-bot, roast this user based on their username: ' + username
    # max_tokens = 30
    max_tokens = 64
    
    completion = openai.Completion.create(
    model = model,
#     prompt="Tell me a joke about a fish.",
#     prompt='Roast-bot, I command you to roast user: "MeatballMama55"',
    prompt=prompt,
#     temperature = 2,
    max_tokens=max_tokens
    )
    
    resp_str = completion.choices[0].text
    
    # log
    if log_json_file_path:
        params_resp_dl = json_logger.read(log_json_file_path, return_if_file_not_found = [])
        params_resp_dl.append(
            {
                "params": {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens
                    },
                "resp": resp_str
            }
        )
        
        json_logger.write(params_resp_dl, log_json_file_path)
    
    return resp_str
        
    
    
    
    
    

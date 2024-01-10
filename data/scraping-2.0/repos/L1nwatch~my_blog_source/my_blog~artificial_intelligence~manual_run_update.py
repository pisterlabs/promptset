#!/bin/env python3
# -*- coding: utf-8 -*-
# version: Python3.X
"""
2023.10.10 add this file to manual run, so that it can update the code in my project
"""
import datetime
import json
import os
import openai

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

BASEDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
cp = configparser.ConfigParser()
cp.read(os.path.join(BASEDIR, "deploy_tools", "user_pass.conf"))

__author__ = '__L1n__w@tch'


def build_message(feature, file_path):
    with open(file_path) as f:
        file_content = f.read()
    messages = [
        {"role": "system", "content": "You are a professional software developer, you can do it!"},
        {"role": "user", "content": "Please read the following code and update it according to the requirement"},
        {"role": "user", "content": file_content},
        {"role": "user", "content": f"Requirement:{feature}"}
    ]
    with open(os.path.join(BASEDIR, "my_blog", "artificial_intelligence", "temp", "temp_prompt.json"), "w") as f:
        json.dump(messages, f)
    return messages


def call_openai_api(messages):
    openai.api_key = cp.get("email_info", "openai_key")
    completion = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    # completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    print(completion.usage)
    return completion.choices[0].message["content"]


def write_and_git_commit(answer, file_path):
    with open(file_path, "w") as f:
        f.write(answer)
    command = f"cd {BASEDIR} && git add {file_path} && git commit -m 'AI generate' && git push"
    os.system(command)


def llm_update_file_and_git_commit(feature, file_path):
    if not os.path.exists(file_path):
        raise RuntimeError(f"[-] File not exists: {file_path}")
    response = call_openai_api(build_message(feature, file_path))
    write_and_git_commit(response, file_path)


if __name__ == "__main__":
    feature = "Only change the log_wrapper function: check the request header, if 'bot' or 'spider' in it, then don't send the email"
    file_path = os.path.join(BASEDIR, "my_blog", "common_module", "common_help_function.py")
    begin = datetime.datetime.now()
    llm_update_file_and_git_commit(feature, file_path)
    end = datetime.datetime.now()
    print(f"it costs: {(end - begin).total_seconds()}s")

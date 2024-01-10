import os
from pathlib import Path
import json 

from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, AIMessage)

def greetings(system_message:str, ABSOLUTE_PATH:str) -> AIMessage:
    cache_path = Path().joinpath(ABSOLUTE_PATH, "config")
    file_path = os.path.join(cache_path, "system_messages.json")

    if not os.path.exists(file_path):
        os.makedirs(cache_path, exist_ok=True)
        os.system(f"touch {file_path}")
        echo_cmd = r"echo {} >> " + file_path
        os.system(echo_cmd)
    
    with open(file_path, "r") as json_file:
        cached_messages = json.load(json_file)

    keys = list(cached_messages.keys())

    if system_message in keys:
        return AIMessage(content=cached_messages.get(system_message))

    else:
        chat = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
        )

        langchain_sys_message = SystemMessage(content=system_message)
        chat_return = chat([langchain_sys_message])
        content = chat_return.content

        cached_messages[system_message] = content

        with open(file_path, "w") as json_write:
            json.dump(cached_messages, json_write)

        return AIMessage(content=content)

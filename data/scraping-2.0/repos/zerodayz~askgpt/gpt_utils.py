import datetime
import os
import re
import uuid

import openai

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
API_KEY_PATH = os.path.join(DIR_PATH, "api_key.txt")
LOG_DIR = os.path.join(DIR_PATH, "logs")
CODE_DIR = os.path.join(DIR_PATH, "code")

gpt_system_role = "system"
gpt_user_role = "user"


def gpt_init():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CODE_DIR, exist_ok=True)


def set_openai_api_key():
    if not os.path.exists(API_KEY_PATH):
        raise FileNotFoundError("api_key.txt file not found.")
    with open(API_KEY_PATH) as f:
        openai.api_key = f.read().strip()


def create_openai_message(role, content):
    return {"role": role, "content": content}


def extract_code_from_response(response_text):
    if "```python" in response_text:
        code = response_text.split("```python")[1]
        code = code.split("```")[0]
        language = "py"
        return code, language
    elif "```go" in response_text:
        code = response_text.split("```go")[1]
        code = code.split("```")[0]
        language = "go"
        return code, language
    elif "``go" in response_text:
        code = response_text.split("``go")[1]
        code = code.split("``")[0]
        language = "go"
        return code, language
    elif response_text.startswith("`") and response_text.endswith("`"):
        language = "sh"
        return response_text[1:-1], language
    language = "txt"
    return response_text, language


def extract_uuid_and_content(content):
    code_uuid = None
    if content.startswith("START_UUID:"):
        code_uuid = content.split(":")[1]
    content = re.sub(r"START_UUID:.*:END_UUID", "", content)
    if not code_uuid:
        code_uuid = str(uuid.uuid4())
    return code_uuid, content


def save_log(log_name, arg, code_uuid, text):
    log_file_path = os.path.join(LOG_DIR, log_name)
    with open(log_file_path, "a") as f:
        f.write("Date: "
                + datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                + "\n")
        f.write("Question: " + arg + "\n")
        f.write("UUID: " + code_uuid + "\n")
        f.write("Answer: \n")
        f.write(text + "\n")


def save_code(code_name, code_uuid, text, lang):
    code_directory = os.path.join(CODE_DIR, code_uuid)
    os.makedirs(code_directory, exist_ok=True)
    with open(os.path.join(code_directory, f"{code_name}.{lang}"), "w") as f:
        f.write(text)

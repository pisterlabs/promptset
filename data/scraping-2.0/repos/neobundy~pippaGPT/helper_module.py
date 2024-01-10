from datetime import datetime
from pathlib import Path
import os
import openai
import json
import uuid
from dotenv import load_dotenv
import logging

import settings

logging.basicConfig(filename=settings.LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def get_current_date_time():
    return datetime.now().strftime("Date: %Y-%m-%d, Day of Week: %A, Time: %H:%M:%S")


def read_text_file(file_path):
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def save_to_text_file(content, file_path):
    file_path = Path(file_path)
    with file_path.open("w", encoding="utf-8") as file:
        file.write(content)


def log(log_message, log_type="info"):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_types = {
        "debug": {"emoji": "🪲", "log_func": logging.debug},
        "error": {"emoji": "‼️", "log_func": logging.error},
        "test": {"emoji": "🛠️", "log_func": logging.info},
        "info": {"emoji": "✏️", "log_func": logging.info},
    }

    log_type = log_type.lower()
    log_data = log_types.get(log_type, log_types["info"])

    log_message = f"{log_data['emoji']} {log_message}"
    print(f"{current_datetime} - {log_message}")
    log_data['log_func'](log_message)


def get_openai_model_names(gpt_only=False):
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    try:
        models = dict(openai.Model.list())
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    if "data" not in models:
        print("No 'data' key found in the models dictionary.")
        return []

    model_names = []
    for m in models["data"]:
        model_id = m.get("id", "")
        if gpt_only:
            if model_id.startswith("gpt"):
                model_names.append(model_id)
        else:
            model_names.append(model_id)

    model_names.sort()
    return model_names


def get_folders(path):
    return sorted(
        [f for f in Path(path).iterdir() if f.is_dir()],
        key=lambda x: os.path.getctime(os.path.join(path, x)),
        reverse=True,
    )


def get_folder_names(path):
    return sorted(
        [f.name for f in Path(path).iterdir() if f.is_dir()],
        key=lambda x: os.path.getctime(os.path.join(path, x)),
        reverse=True,
    )


def get_files(path):
    return sorted(
        [f for f in Path(path).iterdir() if f.is_file()],
        key=lambda x: os.path.getctime(os.path.join(path, x)),
        reverse=True,
    )


def get_filenames(path):
    return sorted(
        [f.name for f in Path(path).iterdir() if f.is_file()],
        key=lambda x: os.path.getctime(os.path.join(path, x)),
        reverse=True,
    )


def make_unique_folder():
    unique_folder_name = str(uuid.uuid4())
    Path(unique_folder_name).mkdir(parents=True, exist_ok=True)
    return unique_folder_name


def make_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_conversation_to_json(messages, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def load_conversation_from_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


sample_messages = [
    {"role": "System", "content": "System's message."},
    {"role": "Human", "content": "Human's message."},
    {"role": "AI", "content": "AI's message."},
]

if __name__ == "__main__":
    load_dotenv()
    log(os.path.basename(__file__), "test")
    model_names = get_openai_model_names()
    for m in model_names:
        print(m)

import openai
import json
import concurrent.futures
import threading
import os
import time
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = os.environ.get("DATA_FOLDER")
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER")
ERROR_FOLDER = os.environ.get("ERROR_FOLDER")
PROMPT_FILE = os.environ.get("PROMPT_FILE")
write_output_lock = threading.Lock()
write_error_lock = threading.Lock()
MAX_WORKERS = 2

PRE_PROMPT = open(PROMPT_FILE).read()

def call_chat_completion_api(data_entry, output_file_path, error_file_path):
    try:
        openai.api_key = os.environ.get("OPENAI_KEY")
        data_entry_string = json.dumps(data_entry)
        responses = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[{"role": "system", "content": PRE_PROMPT},
                      {"role": "user", "content": data_entry_string}],
            n=1,
            temperature=0,
        )
        instruction = responses['choices'][0]['message']['content'].strip()
        data_entry['instruction'] = instruction

        with write_output_lock:
            if not os.path.exists(output_file_path):
                with open(output_file_path, "w") as f:
                    json.dump(data_entry, f)
                    f.write("\n")
            with open(output_file_path, "a") as f:
                json.dump(data_entry, f)
                f.write("\n")
    except Exception as e:
        if "Rate limit reached" in str(e):
            print("Rate limit reached. Sleeping for 60 seconds.")
            time.sleep(65)  # Sleep for 60 seconds
            return call_chat_completion_api(data_entry, output_file_path, error_file_path)
        else:
            print(e)
            if not os.path.exists(error_file_path):
                with open(error_file_path, "w") as f:
                    json.dump(data_entry, f)
                    f.write("\n")
            with write_error_lock:
                with open(error_file_path, "a") as f:
                    json.dump(data_entry, f)
                    f.write("\n")

def process_file(filename, uid):

    os.makedirs(os.path.join(DATA_FOLDER, uid), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, uid), exist_ok=True)
    os.makedirs(os.path.join(ERROR_FOLDER, uid), exist_ok=True)
    
    full_input_path = os.path.join(DATA_FOLDER, uid, filename)
    full_output_path = os.path.join(OUTPUT_FOLDER, uid, "output.json")
    full_error_path = os.path.join(ERROR_FOLDER, uid, filename)
    
    with open(full_input_path, "r") as f:
        print('what is it', f)
        data_list = json.load(f)
        if type(data_list) is not list:
            data_list = [data_list]
        print(data_list)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(lambda data_entry: call_chat_completion_api(data_entry, full_output_path, full_error_path), data_list)

def generate_inst(uid):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    if not os.path.exists(ERROR_FOLDER):
        os.makedirs(ERROR_FOLDER)

    personal_data_dir = os.path.join(DATA_FOLDER, uid)

    filenames = [f for f in os.listdir(personal_data_dir) if f.endswith('.json')]
    for _file in filenames:
        process_file(_file, uid)
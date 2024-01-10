import openai
import time
import json
import os
from configparser import ConfigParser

config_object = ConfigParser()
config_object.read("/home/pi/auto-debater/config.ini")

# Initialize the OpenAI client
os.environ['OPENAI_API_KEY'] = config_object["USERINFO"]['OPENAI_API_KEY']
client = openai.OpenAI()

def get_json(obj):
    json_format = json.loads(obj.model_dump_json())
    return json_format

# Function to create a run and get the latest message
def create_run_and_get_message(thread_id, assistant_id):
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    print(f"Run created with ID: {run.id}")

    # Wait for the run to complete
    while run.status in ["queued", "in_progress"]:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        time.sleep(5)
        print(f"Checking run status: {run.status}")

    # Retrieve the latest message from the thread
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages_json = get_json(messages)
    latest_message = messages_json['data'][0]['content'][0]['text']['value']
    quotes = [annotation['file_citation']['quote'] for annotation in messages_json['data'][0]['content'][0]['text']['annotations'] if 'file_citation' in annotation]
    print(f"Latest message: {latest_message}")
    print(f"Quotes: {quotes}")
    return latest_message, quotes

# IDs of your assistants and the thread
thread = client.beta.threads.create()
get_json(thread)
thread_id = thread.id
print(f"Thread created with ID: {thread_id}")
assistant_ids = ['asst_pISJlFY22I5Ic0bVqZSNrJQk', 'asst_e1m8jjbzvNMEV5A4cQAqndKz']

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Stelt u zichzelf alstublieft voor en noem het onderwerp waarover u wilt beginnen. Houdt het debat alstublieft netjes en respectvol maar wees ook kritisch op elkaar waar nodig.",
)
print(f"Message created with ID: {message.id}")

try:
    with open(f'messages_{thread_id}.txt', 'a') as file:
        while True:
            for assistant_id in assistant_ids:
                latest_message, quotes = create_run_and_get_message(thread_id, assistant_id)
                file.write(f"Message:\n {latest_message}\n")  # Write the message
                for quote in quotes:
                    file.write(f"Quote:\n {quote}\n")  # Write each quote
                file.write("\n")  # Add a newline for separation
                time.sleep(1)  # Optional: Sleep between messages

except KeyboardInterrupt:
    print("Stopping the message collection.")

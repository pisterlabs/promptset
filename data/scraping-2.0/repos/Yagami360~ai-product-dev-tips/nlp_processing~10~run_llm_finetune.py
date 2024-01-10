import time
import json
import io

import openai
from langchain.adapters import openai as openai_adapter
from langchain.load import load
from langsmith import schemas


# def convert_messages(example: schemas.Example) -&gt; dict:
#     messages = load.load(example.inputs)['messages']
#     message_chunk = load.load(example.outputs)['generations'][0]['message']
#     return {"messages": messages + [message_chunk]}

finetuning_messages = openai_adapter.convert_messages_for_finetuning(messages)

my_file = io.BytesIO()
for group in finetuning_messages:
    if any(["function_call" in message for message in group]):
        continue
    my_file.write((json.dumps({"messages": group}) + "\n").encode('utf-8'))

my_file.seek(0)
training_file = openai.File.create(
  file=my_file,
  purpose='fine-tune'
)

# Wait while the file is processed
status = openai.File.retrieve(training_file.id).status
start_time = time.time()
while status != "processed":
    print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = openai.File.retrieve(training_file.id).status
print(f"File {training_file.id} ready after {time.time() - start_time:.2f} seconds.")

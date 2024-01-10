from openai import OpenAI
from decouple import config
from time import sleep
import json

client = OpenAI(
    api_key=config('OPENAI_API_KEY')
)

# Load the dataset
with open("trainData.jsonl", 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")

for message in dataset[0]["messages"]:
    print(message)

def wait_untill_done(job_id):
    events = {}
    while True:
        response = client.fine_tuning.jobs.list_events(id=job_id, limit=10)
        # collect all events
        for event in response["data"]:
            if "data" in event and event["data"]:
                events[event["data"]["step"]] = event["data"]["train_loss"]
                messages = [it["message"] for it in response.data]
        for m in messages:
            if m.startswith("New fine-tuned model created: "):
                return m.split("created: ")[1], events
        sleep(10)


file = client.files.create(
    file=open("trainData.jsonl", "rb"),
    purpose="fine-tune"
)

model = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
)

job = model.id

new_model_name, events = wait_untill_done(job)
with open("result/new_model_name.txt", "w") as fp:
    fp.write(new_model_name)

print(new_model_name)


from openai import OpenAI
from time import sleep
from lang_chain import api_key

client = OpenAI(api_key = api_key)

# Check if the file path is correct and the file exists
training_file_path = "UAMH_training_data.jsonl"

try:
    with open(training_file_path, "rb") as file:
        f = client.files.create(
            file=file,
            purpose="fine-tune"
        )
except FileNotFoundError:
    print("Training file not found.")
    exit()

print(f)

job = client.fine_tuning.jobs.create(
    training_file=f.id,
    model="gpt-3.5-turbo"
)

print(job)

while True:
    res = client.fine_tuning.jobs.retrieve("ftjob-XB5mzfNY2ty4a4SiTYqa0PVR")  # Use client to retrieve job status
    if res.finished_at != None:
        print(res)
        break
    else:
        print(".", end="")
        sleep(100)

# Print the fine-tuned model ID if the job has succeeded
print(res.fine_tuned_model)
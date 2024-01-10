import openai
import json
import csv

# Configuration of the API key
with open("fine_tuning/key.txt", "r") as file:
    api_key = file.read().strip()

openai.api_key = api_key

# Load data from the CSV file
with open("fine_tuning/sadness_test.csv", "r", encoding="utf-8", ) as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Convert the data to the specified format
training_data = []
for row in data:
    score = float(row['Score'])
    classe = int(score * 10)
    prompt = row['Tweet'] + " ->"
    completion = str(classe / 10) + ".\n"
    training_data.append({"prompt": prompt, "completion": completion})

for entry in training_data:
    entry["prompt"] = "In the following tweet, what is the rate of sadness over 10? Here is the tweet: " + entry["prompt"]

# Convert the data to JSONL
file_name = "training_data.jsonl"
with open(file_name, "w") as output_file:
    for entry in training_data:
        json.dump(entry, output_file)
        output_file.write("\n")
        
# Convert the data to the chat-completion format
chat_data = []
for entry in training_data:
    message = {
        "messages": [
            {"role": "system", "content": "You are a sadness rating assistant."},
            {"role": "user", "content": "In the following tweet, what is the rate of sadness over 10? Here is the tweet: " + entry["prompt"]},
            {"role": "assistant", "content": entry["completion"]}
        ]
    }
    chat_data.append(message)

# Convert the chat data to JSONL
file_name = "training_data.jsonl"
with open(file_name, "w") as output_file:
    for entry in chat_data:
        json.dump(entry, output_file)
        output_file.write("\n")

# Upload the data
response = openai.File.create(
  file=open(file_name, "rb"),
  purpose='fine-tune'
)
file_id = response['id']

# Create a fine-tuning job
fine_tuning_response = openai.FineTuningJob.create(
    training_file=file_id, 
    model="gpt-3.5-turbo"
)
job_id = fine_tuning_response['id']

print(f"Fine-tuning job created with ID: {job_id}")

# Note: The fine-tuning process may take some time to complete. 
# After it's done, you can use the fine-tuned model for inferences as shown in the provided documentation.

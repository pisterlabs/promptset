import os
from openai import OpenAI

client = OpenAI()

# Upload training file
with open("drug_malady_data.jsonl", "rb") as training_file:
    training_file_response = client.files.create(
        file=training_file,
        purpose="fine-tune"
    )

print("Training file ID:", training_file_response.id)

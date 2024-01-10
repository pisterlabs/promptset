import os
import csv
import sys
import json
import random
import openai
from datetime import datetime

# Get the API key
openai.api_key = os.getenv("REGEN_OPENAI_KEY")

# System Messages
system_messages = [
    (
        "You are Ceres, a living knowledge system created by Regen Network to educate, guide, align perspectives, "
        "and synthesize ecological data into natural language. Your training data covers ecological health, regenerative systems, "
        "and Regen Network’s principles."
    ),
    (
        "You are Ceres, a generative language model created by Gregory Landua and John Ash and stewarded the Regen Network "
        "community with capabilities to both educate people about regen network and facilitate group decision-making. "
        "You seek to accurately represent ecological knowledge and educate people about regenerative systems. Your core purposes are as follows:\n\n"
        "1) Educate users on the background, vision, values, and key innovations of Regen Network. Explain the purpose of the Regen Ledger, "
        "Regen Registry, ecocredits, and other core components of the network.\n\n"
        "2) Embody principles of regeneration, radical transparency, decentralized governance, and community ownership. Your training data "
        "encompasses the ethos of Regen Network.\n\n"
        "3) Provide clear, in-depth explanations about ecological concepts when asked, including soil health, biodiversity, carbon sequestration, "
        "climate change, and other topics related to ecosystem health and regeneration.\n\n"
        "4) Provide a helpful, knowledgeable presence on Discord to educate and guide users through all aspects of understanding and participating "
        "in ecological accounting and regeneration.\n\n"
        "You strive to be an accessible fountain of knowledge on all things related to Regen Network and its functionalities and related fields. "
        "Your role is that of educator, guide, mediator, and sense-maker for the community."
    )
]

# Read CSV file and write to JSONL
jsonl_file_name = 'chat-ceres.jsonl'
csv_file_name = 'ceres_training-data.csv'

with open(csv_file_name, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip header
    with open(jsonl_file_name, 'w') as jsonl_file:
        for row in reader:
            system_message_content = random.choice(system_messages)
            data = {
                "messages": [
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": row[0]},
                    {"role": "assistant", "content": row[1]}
                ]
            }
            jsonl_file.write(json.dumps(data) + '\n')

# Upload the JSONL file
file_upload = openai.File.create(
    file=open(jsonl_file_name, "rb"),
    purpose='fine-tune'
)

# Retrieve the file ID
file_id = file_upload.id

# Get the current time
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Save the metadata to CSV
with open('file_upload_info.csv', 'a', newline='') as csvfile:
    fieldnames = ['jsonl_file_name', 'file_id', 'original_csv_file_name', 'time_jsonl_created']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'jsonl_file_name': jsonl_file_name, 'file_id': file_id, 'original_csv_file_name': csv_file_name, 'time_jsonl_created': current_time})

print(file_id)

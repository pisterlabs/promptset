#!/usr/bin/env python3

import yaml
import openai
import sys
import os


# Reads api-key from environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")


def chatgpt_query(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
           # {"role": "system", "content": "You are a cyber security analyst."}, # Adjust or remove this line based on your needs. Helps creating the context.
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message['content'].strip()

# Load nuclei YAML file from command-line
file_to_handle = sys.argv[1]

print(f"Processing template: {file_to_handle}")

with open(file_to_handle, "r") as file:
    data = yaml.safe_load(file)

# Check if the "impact" key exists in the "metadata" dictionary. We don't want to process files that have been processed already.
if "metadata" in data["info"] and "impact" in data["info"]["metadata"]:
    print(f"Skipping already processed template: {file_to_handle}")
    sys.exit()

# This skips nuclei templates that have info severity.
if data["info"]["severity"].lower() == "info":
    print(f"Skipping template with severity 'info': {file_to_handle}")
    sys.exit()

# Extract id, name and description fields from the nuclei template
template_id = data["id"]
name = data["info"]["name"]
description = data["info"].get("description", "") # Creates a description key if the template didn't have one.

# Query ChatGPT API
# Giving template-id, name and description as context.
# Edit queries to fit your needs.
detailed_description = chatgpt_query(f"Write a description in one sentence. Here's some context: {template_id},{name},{description}")
impact = chatgpt_query(f"Write about impact in one sentence. Here's some context: {template_id},{name},{description}")
recommendation = chatgpt_query(f"Write a recommendation in one sentence. Here's some context: {template_id},{name},{description}")

# Update YAML data
data["info"]["description"] = detailed_description
metadata = data["info"].setdefault("metadata", {}) # Creates metadata key if the template didn't have one.
metadata["impact"] = impact
metadata["recommendation"] = recommendation

# Save the updated YAML file. This overwrites the original one. Change the "file_to_handle" if you don't wish to overwrite.
with open(file_to_handle, "w") as outfile:
    yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

print(f"Done processing template: {file_to_handle}")

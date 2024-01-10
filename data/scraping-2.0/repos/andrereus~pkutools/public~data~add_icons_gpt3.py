import csv
import openai
import os
from dotenv import load_dotenv
import emoji

load_dotenv()
load_dotenv("../../.env.local")
openai.api_key = os.environ["OPENAI_API_KEY"]

def is_valid_emoji(icon):
    demojized_icon = emoji.demojize(icon)
    return demojized_icon.startswith(":") and demojized_icon.endswith(":")

def find_matching_icon_gpt3(name):
    prompt = f"Given the following food name: '{name}', provide the most suitable food emoji."
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=5,
        n=1,
        stop=None,
        temperature=0,
    )

    icon = response.choices[0].text.strip()
    return icon if is_valid_emoji(icon) else "🌱"

with open("usda.csv", newline='', encoding='utf-8') as input_file, open("usda-icon.csv", "w", newline='', encoding='utf-8') as output_file:
    reader = csv.reader(input_file, delimiter=',', quotechar='"')
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    # Write the header
    header = next(reader)
    header.append("icon")
    writer.writerow(header)
    
    for row in reader:
        icon = find_matching_icon_gpt3(row[0])
        row.append(icon)
        writer.writerow(row)

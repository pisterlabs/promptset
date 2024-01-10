import openai
import sys
import json
import os
import nltk
import requests

if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt', quiet=True)

sku = sys.argv[1]
credentials = {}

creds_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../creds2.txt"
)

with open(creds_file_path) as f:
    locations = []
    current_location = None
    
    with open(creds_file_path) as f:
        contents = f.read()
        print("File Contents:", contents)

        for line in f:
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                current_location = line[1:-1]
            elif line.startswith("["):
                current_location = line[1:]
            elif line.endswith("]"):
                current_location += line[:-1]
                locations.append(current_location)
            elif current_location and " = " in line:
                key, value = line.split(" = ")
                credentials[current_location + "_" + key] = value

#print("Locations:", locations)
for location in locations:
    website = location + ".doap.com"
    openai.api_key = credentials.get(website + "_consumer_key")
    city = credentials.get(website + "_city", "N/A")
    phone = credentials.get(website + "_phone", "N/A")
    print()
    print("Location:", location)
    print(
        f"\nWebsite: {website}\n"
        f"City: {city}\n"
        f"Phone: {phone}\n"
        f"OpenAI Key: {openai.api_key}\n\n"
    )

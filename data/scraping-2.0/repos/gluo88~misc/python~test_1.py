
>[gpt-3.5-turbo - Simple example](#scrollTo=xIqQcXsuS_5U)

>[list directory](#scrollTo=0N4pG_TOnsN-)

>[gpt-3.5-turbo roles system user](#scrollTo=AVeGhQNwmXV0)



# gpt-3.5-turbo - Simple example

Simple example

# Simple example
#!pip install --upgrade openai
#!pip install openai     # this was done on Dec 5, 2023
# Setting environment variables in Google Colab
%env OPENAI_API_KEY = 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

#------
# the following  from   https://github.com/openai/openai-python/tree/main/examples/demo.py
#!/usr/bin/env -S poetry run python
from openai import OpenAI

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI()

# Non-streaming:
print("----- standard request -----")
completion = client.chat.completions.create(
    # model="gpt-4",
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        },
    ],
)
print(completion.choices[0].message.content)

# Streaming:
print("----- streaming request -----")
stream = client.chat.completions.create(
   # model="gpt-4",
   model="gpt-3.5-turbo",
   messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
    stream=True,
)
for chunk in stream:
    if not chunk.choices:
        continue

    print(chunk.choices[0].delta.content, end="")
print()

Misc: export, grep

!export |grep OPENAI_API_KEY


# list directory

import os
def output_files(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate over each file
    for file in files:
        # Check if it is a file (not a directory)
        if os.path.isfile(os.path.join(directory, file)):
            # Output the file name
            print(file)

# Specify the directory path
directory = "/content/sample_data"

# Call the function to output files in the directory
output_files(directory)


#gpt-3.5-turbo roles system user

#!pip install openai
%env OPENAI_API_KEY = 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
!export |grep OPENAI_API_KEY

from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

# vision GPT-4V & gpt-4-vision-preview

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response.choices[0])

# base64 image - local photo - extract text from photo

import base64
import requests

# OpenAI API Key
api_key = "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"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/content/IMG1_receipt.jpg"
#image_path = "/content/IMG2.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
         # "text": "What’s in this image?"
          "text": "this is my document. what are the text in this photo?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())

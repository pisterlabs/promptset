from dotenv import load_dotenv
from openai import OpenAI
import base64
import json
import os
from urllib.parse import urlparse

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# image_local = './Images/Invoice.jpg'
# image_url = f"data:image/jpeg;base64,{encode_image(image_local)}"
image_url = 'https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2021/02/19/ML-1955-2.jpg'

client = OpenAI() #Best practice needs OPENAI_API_KEY environment variable
# client = OpenAI('OpenAI API Key here')

response = client.chat.completions.create(
    model='gpt-4-vision-preview', 
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Return JSON document with data. Only return JSON not other text"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ],
        }
    ],
    max_tokens=1000,
)

# Extract JSON data from the response and remove Markdown formatting
json_string = response.choices[0].message.content
json_string = json_string.replace("```json\n", "").replace("\n```", "")

# Parse the string into a JSON object
json_data = json.loads(json_string)

# filename_without_extension = os.path.splitext(os.path.basename(image_local))[0] #for local image
filename_without_extension = os.path.splitext(os.path.basename(urlparse(image_url).path))[0] #for URL image

# Add .json extension to the filename
json_filename = f"{filename_without_extension}.json"

# Save the JSON data to a file with proper formatting
with open("./data/" + json_filename, 'w') as file:
    json.dump(json_data, file, indent=4)

print(f"JSON data saved to {json_filename}")
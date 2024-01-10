import json
import openai
import csv

# Replace with your GPT-4 API endpoint and API key
GPT4_API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
# Set your OpenAI API key
GPT4_API_KEY = json.load(open("ml/openAIKey.json"))["token"]
openai.api_key = GPT4_API_KEY

# Define the conversation
conversation = [
    {"role": "user", "content": "Provide nutritional information for banana in json file format. Do not include any information that is not appropiate for the json file format"},
]

# Call the model
response = openai.ChatCompletion.create(
    model="gpt-4",  # Use the appropriate model
    messages=conversation,
)

# Extract the nutritional information from the response
nutritional_info = response["choices"][0]["message"]["content"]
json_object = json.loads(nutritional_info)

# Define the output JSON file name
output_file = "nutritional_info.json"

# Write the nutritional information to a JSON file
with open(output_file, "w") as file:
    json.dump(json_object, file, indent=4)

print(f"Nutritional information saved to {output_file}")
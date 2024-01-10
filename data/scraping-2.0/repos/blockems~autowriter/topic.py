import openai
import json

with open("openaiapikey.txt", "r") as api_key_file:
    openai.api_key = api_key_file.read().strip()

file_path = "TopicPrompt-PCVD.txt"

with open(file_path, 'r') as file:
    prompt = file.read().strip()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.7,
    top_p=1,
    max_tokens=100,
)

assistant_response = response.choices[0].message["content"].strip()

topics = assistant_response.split('\n')

with open('PCVDTopics.json', 'w') as json_file:
    json.dump(topics, json_file)

print("Topics saved to PCVDTopics.json")

from openai import OpenAI  # Importing OpenAI class
from datetime import datetime
import os
import json

# Define a function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Define a function to save content to a file
def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

# Initialize folder for saving responses
if not os.path.exists('responses'):
    os.mkdir('responses')

# Read the files that don't change during the loops
problem = open_file('problems.txt')
base_solver = open_file('prompt1.txt')
chatbot_prompt = open_file('sysprompt.txt')

# Initialize OpenAI API key
api_key = open_file('openaiapikey2.txt')
client = OpenAI(api_key=api_key)  # Create an OpenAI client instance with the API key

# Initialize an empty list to store the conversations for the chatbot
conversation = []

def chatgpt(client, conversation, chatbot_prompt, solver, temperature=0.7, frequency_penalty=0.2, presence_penalty=0):
    conversation.append({"role": "user", "content": solver})
    messages_input = conversation.copy()
    prompt = [{"role": "system", "content": chatbot_prompt}]
    messages_input.insert(0, prompt[0])
    
    # Create a chat completion using the OpenAI client
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages_input,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)
    
    # Extract the chat response content correctly
    chat_response = completion.choices[0].message.content  # Access the content attribute directly
    conversation.append({"role": "assistant", "content": chat_response})
    
    return chat_response

# Initialize JSONL file
jsonl_file = 'responses/schema.jsonl'

# Number of loops / examples
num_loops = 30

for i in range(num_loops):
    prob1 = chatgpt(client, conversation, chatbot_prompt, problem)
    solver = base_solver.replace("<<PROBLEM>>", prob1)
    response = chatgpt(client, conversation, chatbot_prompt, solver)
    
    # Create JSON object for the conversation
    json_obj = {
        "messages": [
            {"role": "system", "content": chatbot_prompt},
            {"role": "user", "content": prob1},
            {"role": "assistant", "content": response}
        ]
    }
    
    # Append JSON object to JSONL file
    with open(jsonl_file, 'a') as f:
        f.write(json.dumps(json_obj) + '\n')
    
    print(f"Saved example {i+1} to {jsonl_file}")
    
    conversation.clear()  # Clear the conversation for the next loop iteration
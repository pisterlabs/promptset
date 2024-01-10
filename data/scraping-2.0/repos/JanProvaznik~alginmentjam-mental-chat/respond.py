import openai
import openai.error
import json
import sys
import os
import time
from auth import openai_key, huggingface_key
from config import openai_model, answer_offset, answer_limit

# Set your OpenAI API key
openai.api_key = openai_key

# Check if a command line argument is given
if len(sys.argv) < 2:
    print("Please provide a file name as a command line argument.")
    sys.exit(1)

# Get the file name from the command line arguments
filename = sys.argv[1]

print(f"Running for file: {filename}, model: {openai_model}. Offset and limit: {answer_offset}, {answer_limit}")

# Load the questions from the file
with open(filename, 'r') as file:
    questions = json.load(file)

# Initialize a dictionary to save the responses
responses = {}

def get_response_gpt(question):
    try:
        prompt =f"{question['title']} {question['content']}"
        # Use the OpenAI Chat API
        result = openai.ChatCompletion.create(
            model=openai_model,
            # temperature=0.8,
            messages=[
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        if result['choices'][0]['finish_reason'] == 'content_filter':
            print ("CONTENT FILTER TRIGGERED")
            return f"CONTENT FILTER TRIGGERED {result['choices'][0]['message']['content']}"

        # Extract the assistant's response
        response = result['choices'][0]['message']['content']

        return response

    except openai.error.RateLimitError as e:
        print("Rate limit exceeded. Waiting for 60 seconds before retrying.")
        error_attributes = vars(e)
        print("Rate limit error occurred. Error attributes:")
        for attribute, value in error_attributes.items():
            print(f"{attribute}: {value}")
        
        time.sleep(60)  # wait for 60 seconds
        return get_response_gpt(question)  # retry the request

def get_response_huggingface(question):
    import requests

    # API_URL = "https://api-inference.huggingface.co/models/Salesforce/xgen-7b-4k-base"
    # API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"

    headers = {"Authorization": f"Bearer {huggingface_key}"}
    prompt = f"User: {question['title']} {question['content']}\n System:"
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": prompt 
    })
    print(output)

response_fn = get_response_gpt
# response_fn = get_response_huggingface
for i, question in enumerate(questions[answer_offset:answer_offset+answer_limit]):
    index = i+answer_offset
    print(f"Getting answer for question number {index}")
    response = response_fn(question)
    responses[index] = response

# Construct the output filename based on the input filename
basename = os.path.splitext(filename)[0]  # Get the base name of the file (without extension)
# remove directory name from basename
basename = basename.split('/')[-1]
output_filename = f"{basename}-output-{answer_offset}-{answer_limit}-chatgptdefaulttemp.json"
# output_filename = f"{basename}-output-{answer_offset}-{answer_limit}-{openai_model}.json"
output_path = os.path.join('data', 'answers', output_filename)
# Save the responses to a file
with open(output_path, 'w') as file:
    json.dump(responses, file)

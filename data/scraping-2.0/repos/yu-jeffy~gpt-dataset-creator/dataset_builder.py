from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Create OpenAI client
client = OpenAI()

# prompt gpt-4
completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": ""},
        {
            "role": "user",
            # example is for LLaMa training dataset, modify JSON to match format needed
            # add your desired dataset information to generate also
            # should generate 10 JSON entries every request
            "content": """We are generating a dataset in the EXACT following format, each data entry in the JSON object will be a single string with no line breaks:

                        ```json
                        [
                            {
                             "conversation":"### Human: [Your prompt or question here] ### Assistant: [The assistant's response here] "
                            },
                            {
                             "conversation":"### Human: [Your prompt or question here] ### Assistant: [The assistant's response here] "
                            }, ...
                        ]
                        ```
                        
                        Generate at least 10 entries in the JSON response. Return ONLY the json. """,
        },
    ],
    # set temperature for creativity level
    temperature=0.8,
    max_tokens=4095
)

# print(completion.choices[0].message.content)

# Convert the response to a string
response_str = completion.choices[0].message.content

# Remove Markdown formatting (`response_str` should contain only the JSON array after this operation)
response_str_cleaned = response_str.replace("```json", "").replace("```", "").strip()

# Parse the clean string into a Python object. Produces a list of dictionaries.
# Each dictionary represents a single conversation.
try:
    conversations = json.loads(response_str_cleaned)
except json.JSONDecodeError as e:
    print("Failed to parse JSON:", e)
    # Handle the error further if needed (e.g., log it, return from function, exit script)
    raise

# The file where your JSON data is to be stored
# enter your filename here
file_path = 'FILENAME.jsonl'

# Open the file in append mode, create it if it doesn't exist
with open(file_path, 'a') as file:
    # Iterate over each conversation in the conversations list
    for conversation in conversations:
        # Convert the conversation dictionary to a JSON string
        json_string = json.dumps(conversation)
        
        # Write the JSON string as a new line in the file
        file.write(json_string + '\n')
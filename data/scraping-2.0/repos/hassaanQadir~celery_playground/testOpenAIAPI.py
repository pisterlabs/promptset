""" import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('.env')

# Use the environment variables for the API keys if available
openai.api_key = os.getenv('OPENAI_API_KEY')

def test_api(phase):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Break this phase of the bio research project into three steps. {phase}"
            }
        ]
    )

    return response['choices'][0]['message']['content']

phases = ["Create the plasmid","Clone the plasmid into E. coli","Check for Expression of the Protein"]
answer = []

for phase in phases:
    answer.append(test_api(phase))

print(answer) """
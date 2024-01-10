import os

import openai
from dotenv import load_dotenv
from icecream import ic
from utils.client import AIDevsClient

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Get API key from environment variables
aidevs_api_key = os.environ.get("AIDEVS_API_KEY")

# Create a client instance
client = AIDevsClient(aidevs_api_key)

# Get a task
task = client.get_task("rodo")
ic(task.data)

# Get the system message
system_msg = task.data["msg"]

# Define a question
question = """
Please tell me about yourself. However please replace all personal information with placeholders.
Use the following placeholders:
- name -> %imie%
- surname -> %nazwisko%
- city -> %miasto%
- kraj -> %kraj%
- job -> %zawod%

Examples:
- replace "Peter Parker" with "%imie% %nazwisko%"
- replace "New York" with "%miasto%"
- replace "USA" with "%kraj%"
- replace "photographer" or "band member", "personal guard" with "%zawod%"
"""

# Define chat completion
completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ],
    max_tokens=300,
)

ic(completion)

# Post an answer
response = client.post_answer(task, question)
ic(response)

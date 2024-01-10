import openai
import sys
sys.path.append('../config.py')
from config import OPENAI_API_KEY

# Set up your OpenAI GPT API credentials
openai.api_key = OPENAI_API_KEY

# Define your project or topic of interest
project = "a platform that can help students pursue projects of interest by connecting them with mentors and other students"

# Generate additional information using the GPT API
prompt = f"""I want to build {project}. I want you to scrape the internet and give me any relevant information. The output should be in the format given below in double backticks
``
{{
    "research": [],
    "competitor products": [],
}}
``
"""
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=500,
    n=5,  # Specify the number of completions you want
    stop=None,
)

for i in range(5):
    print(response["choices"][i]["text"])
    print('-'*100)
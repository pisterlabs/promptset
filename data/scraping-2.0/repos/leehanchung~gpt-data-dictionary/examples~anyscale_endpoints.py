import os

import openai
from dotenv import load_dotenv

load_dotenv()

# Using Anyscale Endpoints
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME="codellama/CodeLlama-34b-Instruct-hf"


query = "Translate this variable name 'VAR1' into readable English."


chat_completion = openai.ChatCompletion.create(
    model="codellama/CodeLlama-34b-Instruct-hf",
    messages=[
        {"role": "system", "content": "You are an expert in data management and administration in a large enterprise setting. You have a very deep understanding of data and the common naming schemes of the data and what they means. Please respond the query in a JSON format."}, {"role": "user", "content": query}],
    temperature=0.01,
    stream=True
)

# Streaming
for message in chat_completion:
    message = message["choices"][0]["delta"]
    if "content" in message:
        print(message["content"], end="", flush=True)

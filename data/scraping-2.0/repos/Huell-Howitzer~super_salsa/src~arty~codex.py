import dotenv
import openai
import os
import subprocess

# Set up the OpenAI API client
#openai.api_key = dotenv.get_key(".env", "OPENAI_API_KEY")
openai.api_key = "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"
# Generate code from a natural language prompt
prompt = "Create a function that takes two numbers and returns their sum"
response = openai.Completion.create(
    engine="davinci-003",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.7,
)

# Print the generated code
print(response.choices[0].text)

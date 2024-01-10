import openai
import os

# How to get copilot suggestions?

os.environ["OPENAI_API_KEY"] = "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"

print(os.environ.get("OPENAI_API_KEY"))
# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

print(dir(openai))

print(dir(openai.Model))

openai.Model.list()

# Define the prompt for the code generation
# prompt = """
# Write a function to perform linear regression in Python.
# The function should take two arguments: a dataframe and the name of the target variable.
# It should return the regression coefficients and the R-squared value.
# """
prompt = """What are the top 10 potential use cases for ChatGPT"""

prompt = """How can I use ChatGPT as a personal assistant?"""

prompt = """How to build a custom application that can generate code from a prompt using ChatGPT"""

prompt = """What is the time in Brisbane right now"""

prompt = """How to forecast the the required maintenance for a road network"""

prompt = """Write a Python script to reverse the order of a list"""


# Get the response from ChatGPT
response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=prompt,
    max_tokens=1024,
    n = 1,
    stop=None,
    temperature=0.5,
)

# Build an application that can generate code from a prompt using ChatGPT


# Print the generated code
print(response["choices"][0]["text"])

response.keys()

print(dir(response.id))

dir(response)

response.keys()

#
response['choices'][0].keys()

print(response['choices'])
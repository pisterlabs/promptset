import openai
import argparse
import os

# create the parser
parser = argparse.ArgumentParser()
# add the string argument
parser.add_argument("string_arg", type=str, help="A string argument")
args = parser.parse_args()

# read file.txt content
with open('file.txt', 'r') as f:
    prompt = f.read()

prompt= args.string_arg + prompt
# set the API key
openai.api_key = os.environ.get('API_KEY') 

# define the model and prompt
model_engine = "text-davinci-002"

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0.5,
  max_tokens=2000,
  presence_penalty=0.0
)
# print the response
print(response["choices"][0]["text"])
# print(response)
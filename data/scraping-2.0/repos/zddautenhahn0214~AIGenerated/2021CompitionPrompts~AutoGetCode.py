#purpose of this is to feed it scraped coding prompts and retrieve code automatically from the openAI engine

import os
import openai
import json

# Set up the OpenAI API client
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "OPENAI_API_KEY"


aiEngines = openai.Model.list()

# print(openai.Model.list())

# open a file for writing
with open("engines.txt", "w") as f:
    f.write(json.dumps(aiEngines, indent=4))


# Define the prompt for the code generation task
prompt = "#Return code for a function that takes a string as input and returns the reverse of the string."

# Use the `Completion` endpoint of the GPT-3 API to generate code
# model_engine = "davinci"
# model_engine = "text-davinci-003"

# "Most capable Codex model. Particularly good at translating natural language to code. In addition to completing code, also supports inserting completions within code."
model_engine = "code-davinci-002"
# model_engine = "code-cushman-001"
    
completion = openai.Completion.create(
    model=model_engine,
    prompt=prompt,
    max_tokens=1500,
    # temperature=0,
    # top_p=1,
    # n=1,
    # stream=false,
    # logprobs=null,
    # stop="\n"
    )

# # # generate a response from the model
# # response = openai.Completion.create(
    # {
    # "model": "text-davinci-003",
    # "prompt": "Say this is a test",
    # "max_tokens": 7,
    # "temperature": 0,
    # "top_p": 1,
    # "n": 1,
    # "stream": false,
    # "logprobs": null,
    # "stop": "\n"
    # }
    # )



# Print the generated code
print(completion)

results = completion["choices"][0]["text"]
print(results)


# open a file for writing
with open("answers.txt", "w") as f:
    f.write(results)

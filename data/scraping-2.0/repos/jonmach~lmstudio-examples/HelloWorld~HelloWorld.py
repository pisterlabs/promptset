#
# This is a Hello World for interacting with your local LLM
#

import openai

# Put your URI end point:port here for your local inference server (in LM Studio) 
openai.api_base='http://localhost:1234/v1'
# Put in an empty API Key
openai.api_key=''

# Adjust the following based on the model type
# Alpaca style prompt format:
prefix = "### Instruction:\n" 
suffix = "\n### Response:"

# 'Llama2 Chat' prompt format:
# prefix = "[INST]"
# suffix = "[/INST]"

# This is a simple wrapper function to allow you simplify your prompts
def get_completion(prompt, model="local model", temperature=0.0):
    formatted_prompt = f"{prefix}{prompt}{suffix}"
    messages = [{"role": "user", "content": formatted_prompt}]
    print(f'\nYour prompt: {prompt}\n')
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

prompt = "Hello! Please give me 3 words that rhyme with 'world'"
response = get_completion(prompt, temperature=0)
print(f"LLM's response:{response}")

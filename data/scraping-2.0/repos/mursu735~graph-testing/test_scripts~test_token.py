import tiktoken
import openai
import time
import random
import helpers

# To get the tokeniser corresponding to a specific model in the OpenAI API:

'''
with open("prompt.txt", encoding="utf-8") as file:
    text = file.read()
    encoding = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoding.encode(text))
    print(num_tokens)
    '''

key = ""
with open("key.txt") as file:
    key = file.read()
print(key)
openai.api_key = key

start = time.time()


#instruction = helpers.get_instruction()
instruction = "You are a helpful bot"
#prompt = helpers.get_chapter(1)
prompt = "Tell a joke"

# Check for number of tokens sent in the last minute
whole_prompt = instruction + "\n" + prompt
encoding = tiktoken.encoding_for_model("gpt-4")
num_tokens = len(encoding.encode(whole_prompt))

'''
for i in range(0, 5):
    timediff = time.time() - last_request
    print(f"Time from last message {timediff}")
    time_to_wait = last_request - time.time() + 60
    print(f"Time to wait {time_to_wait}")
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}], temperature=1.0)
    print(chat_completion)
    message = chat_completion.choices[0].message.content
    last_request = time.time()
    #message = "dwsarfe"
    print(message)
    tokens = chat_completion.usage["total_tokens"]
    print(f"Tokens in last message {tokens}")
    time.sleep(3)
'''
chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": instruction}, {"role": "user", "content": prompt}], temperature=1.0)
print(chat_completion)
message = chat_completion.choices[0].message.content
last_request = time.time()
#message = "dwsarfe"
prompt_tokens = chat_completion.usage["prompt_tokens"]
completion_tokens = chat_completion.usage["completion_tokens"]
print(f"Tokens in last message {prompt_tokens}")

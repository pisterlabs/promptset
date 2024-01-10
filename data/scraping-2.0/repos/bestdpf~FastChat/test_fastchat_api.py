import datetime

import openai
openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://202.112.47.151:9999/v1"
openai.api_base = "http://202.112.238.191:9999/v1"

model = "vicuna-wizard-7b"
prompt = "Once upon a time"

# # create a completion
# completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# # print the completion
# print(prompt + completion.choices[0].text)

# create a chat completion

st_time = datetime.datetime.now().timestamp()

completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
end_time = datetime.datetime.now().timestamp()
print(f'cost time is {end_time - st_time}')
# print the completion
print(completion.choices[0].message.content)

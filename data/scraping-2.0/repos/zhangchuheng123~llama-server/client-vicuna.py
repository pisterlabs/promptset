import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:13000/v1"

model = "vicuna-7b-v1.3"
prompt = "HERE COMES THE PROMPTS:"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=1000)
# print the completion
print(prompt + completion.choices[0].text)

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)


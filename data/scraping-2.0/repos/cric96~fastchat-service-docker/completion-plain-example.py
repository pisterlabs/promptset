import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://clusters.almaai.unibo.it:23231/v1"

model = "vicuna"
prompt = "Nel mezzo del cammin di nostra vita" \

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)
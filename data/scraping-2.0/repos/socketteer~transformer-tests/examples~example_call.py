import os
import openai
import math

openai.api_key = os.environ["OPENAI_API_KEY"]

f = open("examples/prompt.txt", "r")
prompt = f.read()

response = openai.Completion.create(
  engine="ada",
  prompt=prompt,
  temperature=0.9,
  max_tokens=1,
  echo=False,
  top_p=1,
  logprobs=0,
  n=1,
)

print(response)

for choice in response.choices:
  print(choice['text'])
# print("PROMPT: ", prompt)
# print("RESPONSE: ", response.choices[0]['text'])
# print("LOGPROBS: ", response.choices[0]["logprobs"]["token_logprobs"], response.choices[0]["logprobs"]["tokens"])
# print("PROBS: ", [math.exp(x) for x in response.choices[0]["logprobs"]["token_logprobs"]])

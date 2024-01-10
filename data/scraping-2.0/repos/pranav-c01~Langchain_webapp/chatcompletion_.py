import os
import openai

openai.api_key = "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"




completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-0301",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)


# with open("x.txt","w") as f:
#     f.write(str(openai.Model.list()))
#     print("done")

# print("\n\n\n\n\n\n\n\n\n\n",openai.Model.list().items())
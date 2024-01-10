import os
import openai

openai.api_key = ""

response = openai.Completion.create(
  model="text-davinci-002",
  prompt="write a python programe to create string encryption and decryption",
  temperature=1,
  max_tokens=4000,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

res = str(response)
t_i = int(res.find("text"))
t_f = int(res.find('''"
    }'''))
t_i = t_i + 8
text_resorces = res[t_i: t_f]

with open("AI_Answer.txt", "a") as f:
  f.write("'''\n\n")
  f.write(text_resorces)
  f.write("\n\n'''")

import openai
import json

with open("api_keys.json", 'r') as f:
  api_keys = json.load(f)
openai.api_key = api_keys["open_AI"]
# with open('key','r') as f:
#   openai.api_key=f.readline()


code_tail = " \n\n\"\"\" Explanation of what the code does"

def get_explanation(file_name):
  with open(file_name, 'r') as f:
      input_code = f.read() + code_tail

  response = openai.Completion.create(
    model="code-davinci-002",
    prompt=input_code,
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\"\"\""]
  )
  print('explanation',response,response["choices"][0].text)
  return response["choices"][0].text

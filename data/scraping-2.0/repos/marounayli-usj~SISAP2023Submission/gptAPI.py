import os
import openai
import gptkey

os.environ["OPENAI_API_KEY"] = gptkey.secret_key
openai.api_key = os.getenv("OPENAI_API_KEY")

def desc_to_dict(description):
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Q: Write me a python dictionary of HTML attributes that corresponds to the following description " + description + " if you cant find any, give me an empty string\n A:",
    temperature=0,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
  )
  desc = response['choices'][0]['text']
  desc= desc.replace("'","\"")
  return desc



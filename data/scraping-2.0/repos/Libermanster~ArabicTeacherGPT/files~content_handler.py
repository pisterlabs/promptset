#this file is for testing the openai api
#i want to send a prompt to the api and get a response
#send a prompt to the api

import os
import openai
#get environment variable
openai.organization = "org-WxlbTUFffJ71K7W5avWdRiI5"
openai.api_key = os.getenv('OPENAI_API_KEY')


#gettext from firstPrompet.txt in string format and save it in _txt in arabic and english
with open("firstPrompet.txt", "r", encoding="utf-8") as f:
    _txt = f.read()




response = openai.Completion.create(
    engine="text-davinci-003",
    prompt = _txt,
    temperature=0.9,
    max_tokens=500
)
response = response.choices[0].text
print(response)

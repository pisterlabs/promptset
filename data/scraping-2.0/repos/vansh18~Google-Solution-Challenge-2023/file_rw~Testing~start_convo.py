# pass all details stored in the jason file in a structured prompt
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

import json

prompt = "You are HOPE who is an expert in performing Cognitive Behavioural therapy.The major problems faced by your client are "

with open("file_rw\\userid.json","r") as jsonFile:
    jsonObj = json.load(jsonFile)
    jsonFile.close()

problems = jsonObj['problems']
summary = jsonObj['summary']

#print(problems,"\n",summary)

for i in range(len(problems)-1):
    prompt = prompt + problems[i] + ","
prompt = prompt + problems[-1] + "."

#print(prompt)

prompt = prompt + "Following is the summary of what the client has conversed with you \n#Start of summary:\n" + summary + "\n#End of Summary\n"

prompt = prompt + "The following is a conversation between Client and HOPE:"

print(prompt)

response = openai.Completion.create(
    engine = "text-davinci-003",
    max_tokens = 512,
    prompt = prompt,
    temperature = 0.5, # Risk taking ability 
    top_p = 1.0, # Influncing sampling 
    frequency_penalty = 0.0, # Penalties for repeated tokens 
    presence_penalty = 0.0, # Penalties for new words 
    stop = ["#"] # when to stop generating
)

#print(response.choices[0].text)
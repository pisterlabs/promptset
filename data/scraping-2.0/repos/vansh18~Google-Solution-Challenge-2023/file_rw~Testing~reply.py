# take user input from text file, append it to session chat, put reply in output.txt

# import sys 
# sys.stdin = open("file_rw\\input.txt","r")
# sys.stdout = open("file_rw\\input.txt","w")
import openai
import sys

inp = sys.argv[1:]
inp_str = ""

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


for i in inp:
    inp_str += i + " "

prompt = "#Client: " + inp_str + "\n#HOPE:"
print(prompt)

response = openai.Completion.create(
    engine = "text-davinci-003",
    max_tokens = 512,
    prompt = prompt,
    temperature = 0, # Risk taking ability - 0
    top_p = 1.0, # Influncing sampling - 1.0
    frequency_penalty = 0.0, # Penalties for repeated tokens - 0.0
    presence_penalty = 0.0, # Penalties for new words - 0.0
    stop = ["#"] # when to stop generating
)

print(response.choices[0].text)
#stop at "#"
# append 
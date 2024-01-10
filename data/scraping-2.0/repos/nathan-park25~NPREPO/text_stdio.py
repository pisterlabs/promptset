import openai

# Set the API key
openai.api_key = "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"

import os,sys 

prompt = sys.stdin.read()

# clear the cmd window

# os.system('cls')


# Generate text
completion = openai.Completion.create(
    engine="text-davinci-003",
    # engine="gpt-4",
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)

# Get the text
text = completion.choices[0].text
# changing color of the text to green 
# print("\033[1;32;40m"+text+"\n")
print(text+"\n")

# put text to windwos Clipboard
import pyperclip
pyperclip.copy(text)

import openai
import os

openai.api_key = "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"

home = os.path.expanduser("~")


pathin = home + '/.config/nvim/selection.txt'
pathout = home + '/.config/nvim/GPT4/suggestion.txt'

def chat_with_gpt3_5(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=150
    )
    return response.choices[0].message['content']

with open(pathin, 'r') as f:
    prompt = f.read()
    f.close()

messages = [
    {"role": "system", "content": "You are code corrector, you are helping a user to correct his code. You never chime in with explanations, but provide ONLY the corrected code. You dont mention what you changed, you dont say hello. You just print the corrected code."},
    {"role": "user", "content": prompt},
]

response = chat_with_gpt3_5(messages)

# try:
#     response = response.split('```')[1]
# except:
#     pass

# remove all trailing line breaks in the response
response = response.rstrip()

print(response)

with (open(pathout, 'w')) as f:
    f.write(response)
    f.close()

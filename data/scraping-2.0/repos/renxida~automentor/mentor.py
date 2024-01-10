#%%

from datetime import datetime
import os
import openai
import json
openai.organization = "org-HRqLWCACWqb2BZgAXGT6UtE8"
openai.api_key=os.environ["OPENAI_API_KEY"]

# %%

messages = [
    {"role": "assistant", "content": "I am your mentor. I excel at asking the right questions one at a time to help you figure out what you could do to communicate your needs most effectively, and I'm also very good at drafting emails, notes, and plans for communicating with busy people."},
]

def chat(msg):
    messages.append({"role": "user", "content": msg})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    messages.append({"role": "assistant", "content": response["choices"][0]["message"]['content']})
    return response["choices"][0]['message']["content"]
    

while True:
    try:
      msg = input("You: ")
    except EOFError:
        break
    if msg == "":
        break

    print("\n\nMentor: ", chat(msg))

# use chat to assign suitable filename
response = chat("I'm ending this conversation and saving the transcript. Summarize the conversation into a filename like `ask_for_internship.txt`")

# make directory for transcripts
if not os.path.exists('transcripts'):
    os.makedirs('transcripts')

try:
    #extract filename quoted by `` in response
    filename = response.split('`')[1]

    # repalce .txt with .json
    filename = filename.replace('.txt', '.json')
except:
    print("Response: ", response)
    print("Couldn't extract filename from response. Using default filename.")
    filename = "transcript.json"

# prepend timestamp
filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{filename}"

# save messages as json
with open("transcripts/"+filename, 'w') as f:
    json.dump(messages, f, indent=4)

print("conversation saved as transcripts/{filename}")
import os
import openai
from rich import print

# have this environment variable set
openai.api_key = os.getenv("OPENAI_API_KEY")

messagesList = []
print("Enter your system prompt context below. As an example it should be something like: \n'you are an experienced frontend developer who cares about readability'")
system_prompt = input("Leave blank for default: ")
if system_prompt == "":
    system_prompt = "you are an experienced frontend developer who cares deeply about code readability"
messagesList.append({"role": "system", "content": system_prompt})
first_prompt = input("Enter your prompt: ")
messagesList.append({"role": "user", "content": first_prompt})

response = openai.ChatCompletion.create(model="gpt-4", messages=messagesList)

resContent = response["choices"][0]["message"]["content"]
print(resContent)
messagesList.append({"role": "assistant", "content": resContent})
while True:
    next_prompt = input("Enter next prompt (q to quit): ")
    if next_prompt == "q":
        import csv
        with open('chatlog.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, messagesList[0].keys())
            writer.writeheader()
            for message in messagesList:
                writer.writerow(message)
        print("Wrote log of chat to `chatlog.csv`")
        exit()
    messagesList.append({"role": "user", "content": str(next_prompt)})
    response = openai.ChatCompletion.create(model="gpt-4", messages=messagesList)
    resContent = response["choices"][0]["message"]["content"]
    print(resContent)
    messagesList.append({"role": "assistant", "content": resContent})
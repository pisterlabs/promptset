from openai import OpenAI



from pathlib import Path
p = Path.cwd()
path_beginning = str(p.home())+'/PycharmProjects/OPENAI/'
path = path_beginning+""

import os
cwd = os.getcwd()


class ChatGPT():
    def __init__(self):
        f = open(path+"account.txt", "r")
        self.client = OpenAI(api_key=f.read())
    def talk(self, text):
        completion = self.client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ])

        return completion.choices[0].message.content


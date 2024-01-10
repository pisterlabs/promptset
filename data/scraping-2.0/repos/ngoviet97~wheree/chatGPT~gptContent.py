from openai import OpenAI
import pandas as pd
import csv
import random

apiList = []
file_path = '~/chatGPT/apis.txt'
with open(file_path, 'r') as file:
    file_contents = file.readlines()
    apiList = file_contents
    apiList = [x.replace("\n","") for x in apiList]

class gptContent:
    models = ['gpt-3.5-turbo-0301','gpt-3.5-turbo','gpt-3.5-turbo-0613','gpt-3.5-turbo-1106','gpt-3.5-turbo-16k','gpt-3.5-turbo-16k-0613']
    def __init__(self):
        api = random.choice(apiList)
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api,
        )

    def command(self,prompt):
        model = random.choice(gptContent.models)

        mainContent = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            model=model,
        )
        return mainContent.choices[0].message.content

import os
import openai
import time
import csv
import json

class EmailGenerator:
    def __init__(self):
        self.init_api()
        self.translate_table = str.maketrans('aeiou', 'pzwbz')
        
    # def init_api(self):
    #     with open(".env") as env:
    #         for line in env:
    #             key, value = line.strip().split("=")
    #             os.environ[key] = value

    #     openai.api_key = os.environ.get("API_KEY")
    #     openai.organization_id = os.environ.get("ORG_ID")
    openai.api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNjg5ODc1NDU5LCJpYXQiOjE2ODk1MTU0NTksImp0aSI6IjE4MjU2NTE5YjE4ZjRhZTM5ZWU0OGM0ZTMyOGEyM2E1IiwidXNlcl9pZCI6MjZ9.r5DnzggenM1AGqPdXAFPo5J8aUf7lekmllkymaPnY70'

    def generate_email(self, prompt):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=1.5,
            max_tokens=300
        )
        return response.choices[0].text.strip()

    def vowel_swapper(self, text):
        return text.translate(self.translate_table)


class EmailManager:
    def __init__(self, generator):
        self.generator = generator

    def generate_emails(self, number: int, prompt: str):
        messages = []
        for i in range(number):
            messages.append(self.generate_and_save_email(i+1, prompt))
        return messages
        
    def generate_and_save_email(self, index, prompt):
        email = self.generator.generate_email(prompt)
        self.save_to_csv(email, index, 'emails')

        msg = {}
        msg['message1'] = f'Generated {index} emails so far'
        
        swapped_email = self.generator.vowel_swapper(email)
        self.save_to_csv(swapped_email, index, 'swappedfolder')
        msg['message2'] = f'Swapped vowels in {index} emails so far'

        return msg

    def save_to_csv(self, email, index, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/email_{index}.csv","w",encoding='UTF-8') as f:
            writer = csv.writer(f, delimiter=",", lineterminator="\n")
            writer.writerow([email])

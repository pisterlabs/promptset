import openai
import json
import os

class Category:
    def __init__(self):
        # Json load
        jsonFile = open('config.json')
        config = json.load(jsonFile)

        # Load environment variables
        envVar = config['open-ai']
        openai.api_key = os.environ.get(envVar)

    def chooseCategory(self, prompt):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt="Categorize this word in one of this(studies, food, clothes, fun, misc) ex:UNAH=studies\n+ " + prompt + "=",
            temperature=0.7,
            max_tokens=3,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        
        response = str(response['choices'][0]['text'])
        response = response.replace(" ", "")
        response = response.lower()
        return response


        
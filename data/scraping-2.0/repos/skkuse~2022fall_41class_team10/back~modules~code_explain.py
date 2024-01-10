import os
import openai
from dotenv import load_dotenv

class Explain:
    def act(text):
        load_dotenv()
        openai.api_key = os.getenv("API_KEY")
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt= text + "\n\"\"\"\nThe code is doing following:\n",
            temperature=0,
            max_tokens=64,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\"\"\""]
        )
        result = response["choices"][0]["text"]
        list = result.split('\n')
        while '' in list:
            list.remove('')
        result = list[0] + '\n' + list[1] + '\n'+ list[2]
        return result
        

#From example of openai(openai.com)
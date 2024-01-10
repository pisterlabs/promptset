from json import load
from flask import request
from flask_restful import Resource
import openai
import os


openai.api_key = os.environ.get("OPENAI_API_KEY")
print(openai.api_key)


class GrammarCorrector(Resource):
    def post(self):
        text = request.data
        text = "Correct this to standard English:\n" + text.decode("utf-8")

        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=text,
            temperature=0,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].text

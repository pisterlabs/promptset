import os
import openai


class DateSearch:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = ""

    '''
    object constructor, sets up all routes and sorted by price

    :param input: the raw input from text field
    '''

    def __init__(self, input):
        self.response = openai.Completion.create(
            model="text-davinci-002",
            prompt="Extract the date from this text:\n\nText: \"I want to travel on April 22, 2021.\"\Date: 2021-04-22\n\nText: \""+input+"\"\nDate:",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"])["choices"][0]["text"].strip()

    def timeStrip(self, fullTime):
        time = fullTime.split(":")
        strTime = time[0] + ":" + time[1]
        return strTime

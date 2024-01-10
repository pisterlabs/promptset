import os
import openai

class AirportSearch:
    # sets up library
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # sets up variables
    __airports = []

    '''
    object constructor, sets up airport

    :param input: the raw input from text field
    '''

    def __init__(self, input):
        autoResponse = openai.Completion.create(
            model="text-davinci-002",
            prompt="Extract the airport codes from this text:\n\nText: \"I want to fly from Los Angeles to Miami.\"\nAirport codes: LAX, MIA\n\nText: \""+input+"\"\nAirport codes:",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"])  # output follows a specific format
        self.__airports = autoResponse["choices"][0]["text"].strip().split(
            ", ")  # get the output as a string

    '''
    getAirports
    returns the airports found
    
    :rtype: str[]
    '''
    def getAirports(self):
        return self.__airports

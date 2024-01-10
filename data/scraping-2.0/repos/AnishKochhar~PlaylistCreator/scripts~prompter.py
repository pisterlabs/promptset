import os
from dotenv import load_dotenv
from openai import OpenAI

class Prompter():

    def __init__(self, playlist):
        load_dotenv()
        self.bot = OpenAI() # defaults to using os environ variables OPENAI_API_KEY

    """ Converts from 'number. [songname] - [artist]' to (songname, artist) """
    def getSongArtist(self, query):
        q = query[query.find('. ')+2:].replace('"', '').split(' - ')
        if (len(q) == 2):
            return (q[0], q[1])
        else:
            return None
        
    """ Creates the appropriate prompt for ChatGPT """
    def systemPrompt(self, prompt_type, prompt):
        return_value = ""
        if prompt_type == "genres":
            return_value = "Please recommend 5 songs in each of the following niche genres: "
            for genre in prompt:
                return_value += "{}, ".format(genre[0])
        if prompt_type == "artist":
            return_value = "Please recommend 5 songs from artists very similar to {0}".format(prompt)
        if prompt_type == "world":
            return_value = "Please recommend 5 songs from {0} in the {1}".format(prompt[0], prompt[1])
        return return_value
    

    def ask_chatgpt(self, prompt_type, prompt):
        """
        prompt_type=world -> prompt=[country, decade]
        prompt_type=artist -> prompt=artist_name
        prompt_type=genres -> prompt=[(name, description)]
        """
        print("Asking ChatGPT...")
        system_prompt = self.systemPrompt(prompt_type, prompt)
        completion = self.bot.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a music recommendation assistant, skilled in finding songs to match my taste.\
                    I will give you specific prompts and I would like you to return your response in the form [songname] - [artist]."},
                {"role": "user", "content": system_prompt }
            ]
        )
        response = []
        for choice in completion.choices:
            message = choice.message.content
            for query in message.split("\n\n"):
                for q in query[query.find('1.'):].split('\n'):
                    song = self.getSongArtist(q)
                    if (song):
                        response.append(song)
            print("Response: ", response)
        return response
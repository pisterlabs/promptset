
import os
import openai

from text.TextMaker import TextMaker

openai.api_key = os.getenv("OPENAI_API_KEY")

class Gpt35TextMaker(TextMaker):

    def make_text(self, prompt_dict:dict):
        completion = openai.ChatCompletion.create( 
            model = 'gpt-3.5-turbo',
            messages = [ 
                {'role': 'user', 'content': prompt_dict['positive_prompt_text']}
            ],
            temperature = 0.6
            # temperature = 0.7
            # temperature = 0.55

        )
        return completion['choices'][0]['message']['content']


# def generate_prompt(animal):
#     return """Suggest three names for an animal that is a superhero.

# Animal: Cat
# Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
# Animal: Dog
# Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
# Animal: {}
# Names:""".format(
#         animal.capitalize()
#     )
    

import openai
from decouple import config

SCRIPT = """
Caso haja erros no texto abaixo corrija, caso contrário apenas ignore; texto:

"""

openai.api_key = config('API_KEY')

def openai_correction(text:str):
    """
    Correct text using gpt inteligence
    """
    text_solution = SCRIPT + text
    
    response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=text_solution,
            max_tokens=100,
            temperature=0.7,
            n=1,
            stop=None,
        )
    return response.choices[0].text.strip()

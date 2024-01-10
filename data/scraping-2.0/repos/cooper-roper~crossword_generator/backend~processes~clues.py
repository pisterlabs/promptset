import openai
import os
from dotenv import load_dotenv

def make_clues(crossword_words, topic):
    
    load_dotenv()
    
    AI_key = os.getenv("OpenAI_API_KEY" )
    openai.api_key = AI_key
    
    system = {"role": "system", "content": f"You are a crossword puzzle maker. You need to clues for the crossword with the theme of {topic}. Do not provide the length of the word."}
    clues = []
    
    for word in crossword_words:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[
                system,
                {"role": "user", "content": f"Generate a clue for the word {word}."}
            ]
        )
        clues.append({word: response["choices"][0]["message"]["content"]})
    
    return clues
    
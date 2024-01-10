import openai
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY


# Генерация описания с помощью OpenAI API
def generate_title(title: str) -> str:
    prompt = f"choose a category from this list - [family, sport, nfactorial incubator, education, housework, work, health, hobby, holidays, shopping, fashion, books] where each category separated by a comma, that can match this task/note: {title}.)"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    title = response.choices[0].text.strip()
    return title

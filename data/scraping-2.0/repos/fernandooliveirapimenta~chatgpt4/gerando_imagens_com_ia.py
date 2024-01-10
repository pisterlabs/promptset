import os
import openai

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY") #essa chave deve ser gerada previamente no site da openai
    resultado = openai.Image.create(
        prompt="A cat sitting on a couch.",
        n=2,
        max_width=400)
    print(resultado)

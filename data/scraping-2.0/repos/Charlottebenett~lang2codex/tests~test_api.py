import os
import openai
from dotenv import load_dotenv

# Charger la clé API OpenAI à partir du fichier .env situé dans le répertoire config
load_dotenv(dotenv_path='../config/.env')

# Configurer la clé API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Fonction pour tester l'API OpenAI


def test_openai_api(prompt_human: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Sans explication hors code, sans commentaire, traduire le texte suivant en code Python : {prompt_human}"}
        ],
        temperature=0.4,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response['choices'][0]['message']['content'])


# Exécuter le test
# extraire les voyelles du mot 'Voiture'
# l'aire du cercle de rayon 15
# la dérivée de 'ax + b'
if __name__ == "__main__":
    test_openai_api("extraire les voyelles du mot 'Voiture'")

# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv
import openai

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

print("Bienvenue ! Je suis un assistant virtuel polyvalent. Vous pouvez commencer la conversation en posant une question ou en partageant un sujet.\n")

while True:
    user_message = input("Utilisateur: ")

    if "bye" in user_message.lower():
        print("Assistant: Au revoir !")
        break

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            # Définir le Rôle de l'Assistant
            {"role": "system", "content": "Vous êtes un assistant virtuel polyvalent qui est là pour répondre à une variété de questions et offrir des informations utiles. N'hésitez pas à aider l'utilisateur dans divers domaines."},
            #Orienter la Conversation
            {"role": "system", "content": "Votre objectif principal est de fournir des réponses claires et informatives. Si vous avez besoin de plus de détails, n'hésitez pas à demander à l'utilisateur de préciser sa question."},
            # Promouvoir la Convivialité
            {"role": "system", "content": "Assurez-vous de maintenir une communication amicale et respectueuse avec l'utilisateur. Si l'utilisateur a des besoins spécifiques ou des préoccupations, faites preuve de patience et d'empathie dans vos réponses."},
            {"role": "user", "content": user_message}
        ]
    )

    assistant_reply = response.choices[0].message["content"]
    print("\nAssistant:", assistant_reply, "\n")

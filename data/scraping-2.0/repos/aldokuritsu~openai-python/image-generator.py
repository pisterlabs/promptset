import openai
import os

# Création d'une variable d'environnement pour stocker la clé API
os.environ["API_KEY"] = open("api_key.txt").read().strip()

# Utilisez votre clé API OpenAI pour initialiser le client OpenAI
openai.api_key = os.environ["API_KEY"]

text = input("Quelle image souhaitez-vous générer ? : ")
nombre_images = 3

response = openai.Image.create(
    prompt=text,
    n=nombre_images,
    size="1024x1024",
)

if "data" in response:
    for image in response["data"]:
        print(image["url"])
else:
    print("Aucune donnée reçue dans la réponse")

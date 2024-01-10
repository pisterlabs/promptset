import sys
import openai
import webbrowser
import requests
import random

openai.api_key="    "

def img_gen(query):
    response = openai.Image.create(
        prompt=query,
        n=1,
        size="1024x1024"
    )
    return response['data'][0]['url']

# programme principal
exit_conditions = ("exit")
get_conditions = ("get")
print_conditions = ("print")
open_conditions = ("open")
new_conditions = ("new")

while True:
    query= input("Entrez les mots clés souhaités séparés par une virgule pour générer l'image, ou tapez 'exit' suivi de ENTER pour quitter : ")
    if query in exit_conditions:
        sys.exit()

    url = img_gen(query)

    while True:
        query = input("Tapez 'open' suivi de ENTER pour afficher l'image dans le navigateur, 'get' suivi de ENTER pour télécharger l'image générée, 'print' suivi de ENTER pour afficher l'URL, 'new' suivi de ENTER pour générer une nouvelle image ou ' exit' suivi de ENTER pour quitter :")
        if query in exit_conditions:
            sys.exit()
        if query in get_conditions:
            random_number = str(random.randint(0,1000)) # générer un nombre aléatoire entre 0 et 1000
            filename = "picgpt" + random_number + ".png" # créer un nom de fichier avec un nombre aléatoire
            response = requests.get(url)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print("Image téléchargée en tant que", filename)
        if query in print_conditions:
            print(url)
        if query in open_conditions:
            webbrowser.open(url)
        if query in new_conditions:
            break
        else:
            print("    ")

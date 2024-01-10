""" File to call chatGPT4 to make summary of a script """
from token_counter import get_number_of_tokens
from openai import OpenAI
import datetime

client = OpenAI()

file_name = "./transcriptions/EPR-28-S5-1-TriadeSante.txt"


def read_file(file_name):
    with open(file_name) as f:
        return f.read()


def save_file(file_name, text):
    with open(file_name, "w") as f:
        f.write(text)


def call_chat_gpt(
    model=["gpt-4-32k", "gpt-4-0613", "gpt-4-1106-preview", "gpt-3.5-turbo-0301"][2],
    prompt="",
):
    # max_tokens = 4096
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        # temperature=0,
        # max_tokens=max_tokens,
        n=1,
    )

    return completion.choices[0].message.content


def analyze_num_tokens(gpt_input, is_GPT35Turbo=False):
    num_tok = get_number_of_tokens(gpt_input)
    if is_GPT35Turbo:
        cost = num_tok * 0.000001
    else:
        cost = num_tok * 0.00001
    print(f"Nombre de tokens: {num_tok}, estimated cost is {cost} $")


ma_prompt = f"""
Tu es un assistant scientifique qui me prépare une note extrèmement detaillé au format markdown texte fourni.
Avec cette note, je comprends en detail les conseils donnés dans le script et je peux tous les appliquer dans ma vie de tous les jours, sans exceptions.

- Rédige un résumé très très détaillé, complet, approfondi.
- S'appuyer strictement sur le texte fourni, sans inclure d'informations externes.
- Formate le résumé sous forme de section avec différents niveaux de titres. Pas de liste à puces, mais du texte détaillé à la place.
- Soit le plus précis possible, ne pas hésiter à ajouter des détails. Recopie les exemples à partir du script afin que je puisse le reproduire dans ma vie.
- Le texte de sortie doit être très long et structuré. C'est important qu'il soit long.


Voici un exemple du format de la sortie attendue. L'exemple est structuré et détaillé, il explique en détails les éléments du script : 

# Titre global du script tel que donné en introduction :  "Nom du script"

## Introduction

Introduction du script.

## Titre de la partie 1 du script 

### 1) Sous partie de la partie 1 du script

Texte detaillé expliquant la sous partie 1 du script en detail. Cette partie n'utilise pas de bullet points, à part dans l'introduction, mais du texte détaillé à la place. Les chiffres donnés par le scipt, les études et les exemples sont cités de mainère tres detaille. 

### 2) Sous partie de la partie 1 du script

Texte detaillé expliquant la sous partie 2 du script en detail. Cette partie n'utilise pas de bullet points, à part dans l'introduction, mais du texte détaillé à la place. Les chiffres donnés par le scipt, les études et les exemples sont cités de mainère tres detaille.

[ un maximum de sous-partie ici ]  

## Titre de la partie 2 du script 

### 1) Sous partie de la partie 2 du script

....

Important ! :  Le réponse comporte le maximum d'éléments du script possible, et suit l'ordre du script. 

Fin de l'instruction. Voici le script : """

# ======================= MAIN =======================

if __name__ == "__main__":
    script = read_file(file_name)

    gpt_input = ma_prompt + "Nom du script : /n " + file_name + " /n Script : " + script

    analyze_num_tokens(gpt_input)

    result = call_chat_gpt(model="gpt-4-1106-preview", prompt=gpt_input)

    # sauvegarde la sortie dans un fichier dans le dossier resultat_final. Le nom du fichier depend de la date et de l'heure
    now = datetime.datetime.now()
    with open(f"./resultat_final/{now.strftime('%Y-%m-%d_%H-%M-%S')}.md", "w") as file:
        file.write(+"\n " + ma_prompt)

    print("resultat sauvegardé dans le dossier resultat_final")

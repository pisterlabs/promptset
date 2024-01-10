import json
import openai
from time import sleep
import os

openai.organization = os.environ.get("OPENAI_ORG_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

file = open("splitting_examples.json", "r")
data = json.load(file)
file.close()

exemple= data["0"]
paragraphs = exemple["text"]
base = exemple["base"]

exemple_incomplet = "onze mai mil neuf cent quinze, vingt-quatre ans, domiciliée à Paris, 7, rue des Prêcheurs *\nfille de Charles Célestin Edouard LEPAGE, décédé, t de Pauline Désirée Germain VAPPEREAU * sa\nveuve * matelassière * domiciliée 7, rue des Prêcheurs, d'autre part ;- Les futurs époux déclarent\nqu'il n'a pas été fait de contrat de mariage .- Marius Fernand VIDAL et Charlotte Lucie LEPAGE ont\ndécalé l'un après l'autre vouloir se prendre pour époux et Nous avons prononcé au nom de la\nloi qu'ils sont unis par le mariage .- En présence de: Charles GAUTHIER * employé, 4, Place du\nLouvre et de Pauline LEPAGE, matelassière * 7, rue des Prêcheurs * témoins majeurs, qui, lecture\nfaite, ont signé avec les époux et Nous, Charles Louis TOURY * Officier de l'état-civil du premier\narrondissement de Paris, Chevalier de la Légion d'Honneur ./."
paragraphes_incomplets = {
    "p1": "",
    "p2": "",
    "p3": "onze mai mil neuf cent quinze, vingt-quatre ans, domiciliée à Paris, 7, rue des Prêcheurs *\nfille de Charles Célestin Edouard LEPAGE, décédé, t de Pauline Désirée Germain VAPPEREAU * sa\nveuve * matelassière * domiciliée 7, rue des Prêcheurs, d'autre part ;-",
    "p4": "Les futurs époux déclarent\nqu'il n'a pas été fait de contrat de mariage .- Marius Fernand VIDAL et Charlotte Lucie LEPAGE ont\ndécalé l'un après l'autre vouloir se prendre pour époux et Nous avons prononcé au nom de la\nloi qu'ils sont unis par le mariage .-",
    "p5": "En présence de: Charles GAUTHIER * employé, 4, Place du\nLouvre et de Pauline LEPAGE, matelassière * 7, rue des Prêcheurs * témoins majeurs, qui, lecture\nfaite, ont signé avec les époux et Nous, Charles Louis TOURY * Officier de l'état-civil du premier\narrondissement de Paris, Chevalier de la Légion d'Honneur ./."
}


def split_text(text_to_split):

    # prompt = ""
    # prompt += "Tâche : Séparer le texte en 5 paragraphes. Le texte donné peut être incomplet ou tronqué, n'inclure que les paragraphes complets.\n p1 = Date et heure maison commune\n p2 = Le mari\n p3 = La mariée\n p4 = acte de mariage + uni\n p5 = Témoin et maire\n\n"
    # prompt += "Exemple:\n"
    # prompt += base
    # prompt += "\nParagraphes:\n"
    # prompt += json.dumps(paragraphs, indent=4)
    # prompt += "\n\n"
    # prompt += "Exemple incomplet:\n"
    # prompt += exemple_incomplet
    # prompt += "\nParagraphes:\n"
    # prompt += json.dumps(paragraphes_incomplets, indent=4)
    # prompt += "\n\n"
    # prompt += "Texte à séparer:\n"
    # prompt += text_to_split
    # prompt += "\n\n"
    # prompt += "Paragraphes :\n"

    for i in range(3):
        for i in range(3):
            try:
                # completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                # temperature=1,
                # messages=[
                #     {"role": "user", "content": prompt,},
                # ]
                # )

                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "You split the given texts into paragraphs. You are aware the text can be incomplete or truncated. The parapgrahs are : p1 = Date et heure maison commune\n p2 = Le mari\n p3 = La mariée\n p4 = acte de mariage + uni\n p5 = Témoin et maire\n\n"},
                    {"role": "user", "content" : base},
                    {"role": "assistant", "content": json.dumps(paragraphs, indent=4)},
                    {"role": "user", "content": exemple_incomplet},
                    {"role": "assistant", "content": json.dumps(paragraphes_incomplets, indent=4)},
                    {"role": "user", "content" : text_to_split}
                ]
                )


                break
            except:
                completion = None
                print("Error while getting answer. Retry in 5 seconds")
                sleep(5)
                continue

        if completion is None:
            print("Error while getting answer. Returning...")

        #check if keys are p1, p2, p3, p4, p5 and if values are not empty or ''
        answer = completion.choices[0].message['content']
        answer = answer.replace('\n', '').replace('.','')
        #remove quote around comma
        answer = answer.replace('\',', '",')
        answer = answer.replace(',\'', ',"')
        #remove quote around space
        answer = answer.replace(' \'', ' "')
        answer = answer.replace('\' ', '" ')
        #remove quote around colon
        answer = answer.replace('\':', '":')
        answer = answer.replace(':\'', ':"')
        #remove quote around {}
        answer = answer.replace('{\'', '{"')
        answer = answer.replace('\'}', '"}')
        #remove \n and -\n
        answer = answer.replace('-\\n', '')
        answer = answer.replace('\\n', ' ')
        #remplacer les apostrophes par des guillemets
        answer = answer.replace("\\'", "\'")
        #print(answer)
        answer = answer[answer.index('{'):]
        #print(f'answer : {answer}')
        answer = json.loads(answer)
        print(answer)

        #check if keys are p1, p2, p3, p4, p5 and if values are not empty or '' :
        if 'p1' not in answer.keys() or 'p2' not in answer.keys() or 'p3' not in answer.keys() or 'p4' not in answer.keys() or 'p5' not in answer.keys():
            print('At least one paragraph is missing... retrying')
            sleep(5)
            continue

        # if answer['p1'] == '' or answer['p2'] == '' or answer['p3'] == '' or answer['p4'] == '' or answer['p5'] == '':
        #     print('At least one paragraph is empty... retrying')
        #     sleep(5)
        #     continue
            
        break

    return answer


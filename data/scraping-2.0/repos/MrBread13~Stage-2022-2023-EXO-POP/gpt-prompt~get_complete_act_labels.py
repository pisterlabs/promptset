import json
import openai
import Levenshtein
from time import sleep
import os

openai.organization = os.environ.get("OPENAI_ORG_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

#intro sequence:
intro = "Nous allons te fournir un certificat de mariage, un document ayant toujours la même mise en forme.Tu vas devoir procéder à l’extraction de certaines données sur plusieurs certificats ensuite. Voici le premier certificat, je précise qu’il est extrait d’un document au format Json et que tu auras toutes les réponses fournies à la fin, cela te permettra de mieux reconnaître ce qu’il te faut obtenir dans les contrats suivants. "
# Un exemple d'acte:
act_example = """Le huit avril mil neuf cent quarante, onze heures cinq minutes ****\ndevant Nous ont comparu publiquement en la maison commune: Camille Marcel MOUROT, barman, né à Dijon\nCôte d'Or * le dix-huit février mil neuf cent onze * vingt-neuf ans, domicilié à Paris, 17, rue\nPierre Lescot, actuellement aux armées; fils de Emile MOUROT et de Pauline MONIOT, époux décédés,\nd'une part ,/- ET Jeanne BLONDEAU, sans profession, née à Jars * Cher * le seize mars mil neuf cent\nneuf, trente et un ans, domiciliée à Paris, 17, rue Pierre Lescot; fille de Emile Joseph BLONDEAU\net de Marie Louise MELOT, son épouse, marchands de bestiaux, domiciliés à Vailly sur Saône * Cher *\nDivorcée de William Louis Paul BERNON, d'autre part ;- Les futurs époux déclarent quil n'a pas été\nfait de contrat de mariage .- Camille Marcel MOUROT et Jeanne BLONDEAU ont déclaré l'un après l'autre\nvouloir se prendre pour époux et Nous avons prononcé au nom de la loi qu'ils sont unis par le mariage.\nEn présence de: Emile SOLIGNAC, commerçant, Médaillé militaire, Croix de Guerre, 16 bis, rue\nLauriston, et de Marcelle MOUROT, vendeuse, 22, rue de l'Echiquier, témoins majeurs, qui, lecture\nfaite ont signé avec les époux et Nous, Pierre Louis Adolphe BERTRAND, Maire du Premier arrondisse-\nment de Paris, Chevalier de la Légion d'Honneur ./ \n  
            {"Jour-mariage": "huit",
            "Mois-mariage": "avril",
            "Annee-mariage": "mil neuf cent quarante",
            "Heure-mariage": "onze heures",
            "Minute-mariage": "cinq minutes",
            "Prenom-adjoint-maire": "Pierre Louis Adolphe",
            "Nom-adjoint-maire": "BERTRAND",
            "Ville-mariage": "Premier arrondisse-\nment de Paris",
            "Prenom-mari": "Camille Marcel",
            "Nom-mari": "MOUROT",
            "Profession-mari": "barman",
            "Ville-naissance-mari": "Dijon\nCôte d'Or",
            "Departement-naissance-mari": "",
            "Pays-naissance-mari": "",
            "Jour-naissance-mari": "dix-huit",
            "Mois-naissance-mari": "février",
            "Annee-naissance-mari": "mil neuf cent onze",
            "Age-mari": "vingt-neuf ans",
            "Ville-residence-mari": "Paris",
            "Departement-residence-mari": "",
            "Pays-residence-mari": "",
            "Numero-rue-residence-mari": "17",
            "Type-rue-residence-mari": "rue",
            "Nom-rue-residence-mari": "Pierre Lescot",
            "Prenom-pere-mari": "Emile",
            "Nom-pere-mari": "MOUROT",
            "Profession-pere-mari": "",
            "Ville-residence-pere-mari": "",
            "Departement-residence-pere-mari": "",
            "Numero-residence-pere-mari": "",
            "Type-rue-residence-pere-mari": "",
            "Nom-rue-residence-pere-mari": "",
            "Prenom-mere-mari": "Pauline",
            "Nom-mere-mari": "MONIOT",
            "Profession-mere-mari": "",
            "Ville-residence-mere-mari": "",
            "Departement-residence-mere-mari": "",
            "Pays-residence-mere-mari": "",
            "Numero-rue-residence-mere-mari": "",
            "Type-rue-residence-mere-mari": "",
            "Nom-rue-residence-mere-mari": "",
            "Prenom-mariee": "Jeanne",
            "Nom-mariee": "BLONDEAU",
            "Profession-mariee": "sans profession",
            "Ville-naissance-mariee": "Jars",
            "Departement-naissance-mariee": "Cher",
            "Pays-naissance-mariee": "",
            "Jour-naissance-mariee": "seize",
            "Mois-naissance-mariee": "mars",
            "Annee-naissance-mariee": "mil neuf cent\nneuf",
            "Age-mariee": "trente et un ans",
            "Ville-residence-mariee": "Paris",
            "Departement-residence-mariee": "Cher",
            "Pays-residence-mariee": "",
            "Numero-rue-residence-mariee": "17",
            "Type-rue-residence-mariee": "rue",
            "Nom-rue-residence-mariee": "Pierre Lescot",
            "Prenom-pere-mariee": "Emile Joseph",
            "Nom-pere-mariee": "BLONDEAU",
            "Profession-pere-mariee": "",
            "Ville-residence-pere-mariee": "",
            "Departement-residence-pere-mariee": "",
            "Numero-residence-pere-mariee": "",
            "Type-rue-residence-pere-mariee": "",
            "Nom-rue-residence-pere-mariee": "",
            "Prenom-mere-mariee": "Marie Louise",
            "Nom-mere-mariee": "MELOT",
            "Profession-mere-mariee": "marchands de bestiaux",
            "Ville-residence-mere-mariee": "",
            "Departement-residence-mere-mariee": "",
            "Pays-residence-mere-mariee": "",
            "Numero-rue-residence-mere-mariee": "",
            "Type-rue-residence-mere-mariee": "",
            "Nom-rue-residence-mere-mariee": "",
            "Prenom-ex-epoux": "William Louis Paul",
            "Nom-ex-epoux": "BERNON",
            "Prenom-temoin-0": "Emile",
            "Nom-temoin-0": "SOLIGNAC",
            "Profession-temoin-0": "commerçant",
            "Age-temoin-0": "",
            "Numero-rue-residence-temoin-0": "16 bis",
            "Type-rue-residence-temoin-0": "rue",
            "Nom-rue-residence-temoin-0": "Lauriston",
            "Ville-residence-temoin": "",
            "Departement-residence-temoin": "",
            "Prenom-temoin-1": "Marcelle",
            "Nom-temoin-1": "MOUROT",
            "Profession-temoin-1": "vendeuse",
            "Numero-rue-residence-temoin-1": "22",
            "Type-rue-residence-temoin-1": "rue",
            "Nom-rue-residence-temoin-1": "de l'Echiquier",
            "Nom-ex-epouse" :"",
            "Prenom-ex-epouse" :""
        }
            """
            

question=""" Maintenant, voici un autre certificat de mariage : je veux que tu m'extrais des données sous la meme forme que les réponses que je t'ai fourni. Précision : compte les arrondissement comme une ville."""

def labels_from_act(act_text : str) -> dict :
    """
    Extrait les labels d'un acte de mariage.
    input: act_text: texte de l'acte de mariage
    output: dictionnaire des label
    """
    prompt= intro + act_example + question + act_text
    #try to get an answer and catch the error if the model doesn't answer or answer with an error. Retry 3 times
    for i in range(3):
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.8,
            messages=[
                {"role": "user", "content": prompt,},
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
        return None
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
    #replace Prenom-du-maire with Prenom-adjoint-maire
    #answer = answer.replace('Prenom-maire', 'Prenom-adjoint-maire')
    #replace Nom-du-maire with Nom-adjoint-maire
    #answer = answer.replace('Nom-maire', 'Nom-adjoint-maire')
    #remplacer les apostrophes par des guillemets
    answer = answer.replace("\\'", "\'")
    #print(answer)
    answer = answer[answer.index('{'):]
    #print(f'answer : {answer}')
    answer = json.loads(answer)


    return answer

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print('begin')
    act_text = """ neuf avril mil neuf cent quarante * onze heures vingt minutes ****\ndevant Nous ont comparu publiquement en la maison commune: Antoine POCARD, porteur aux halles, né à\nParis, dixième arrondissement, le cinq février mil neuf cent un, trente-neuf ans, domicilié à Paris\n56, rue Saint Honoré; actuellement mobilisé- fils de Jeanne Marie POCARD- Veuf de Juliette **\nAlbertine GAYRARD, d'une part ,/- ET Adrienne Jeanne ALBRIEUX, journalière, née à Paris, onzième\narrondissement, le onze septembre mil neuf cent deux, trente-sept ans; domiciliée à Paris, 56, rue\nSaint Honoré; fille de Marie Charles ALBRIEUX, sans profession, domicilié à Aulnay sous Bois * Sein\net Oise * et de Marguerite TERLES, décédée, d'autre part ;- Les futurs époux déclarent qu'il n'a\npas été fait de contrat de mariage .- Antoine POCARD et Adrienne Jeanne ALBRIEUX ont déclaré l'un\naprès l'autre vouloir se prendre pour époux et Nous avons prononcé au nom de la loi qu'ils sont\nunis par le mariage .- En présence de: Fernande PASTEAU, concierge, 13, rue Dussoubs, et de\nAmélie MASSONIE, ménagère, 10 rue Volta, témoins majeurs, qui, lecture farte ont signé avec\nles époux et Nous, Charles Louis TOURY, Officier de l'état-civil du premier arrondissement de\nParis, Chevalier de la Légion d'Honneur ./"""

    labels = labels_from_act(act_text)
    print('end')

    # On vérifie que les labels extraits sont bien les mêmes que ceux de l'acte de mariage:

    ref = {
            "Jour-mariage": "neuf",
            "Mois-mariage": "avril",
            "Annee-mariage": "mil neuf cent quarante",
            "Heure-mariage": "onze heures",
            "Minute-mariage": "vingt minutes",
            "Prenom-adjoint-maire": "Charles Louis",
            "Nom-adjoint-maire": "TOURY",
            "Ville-mariage": "premier arrondissement de\nParis",
            "Prenom-mari": "Antoine",
            "Nom-mari": "POCARD",
            "Profession-mari": "porteur aux halles",
            "Ville-naissance-mari": "Paris, dixième arrondissement",
            "Departement-naissance-mari": "",
            "Pays-naissance-mari": "",
            "Jour-naissance-mari": "cinq",
            "Mois-naissance-mari": "février",
            "Annee-naissance-mari": "mil neuf cent un",
            "Age-mari": "trente-neuf ans",
            "Ville-residence-mari": "Paris",
            "Departement-residence-mari": "",
            "Pays-residence-mari": "",
            "Numero-rue-residence-mari": "56",
            "Type-rue-residence-mari": "rue",
            "Nom-rue-residence-mari": "Saint Honoré",
            "Prenom-pere-mari": "",
            "Nom-pere-mari": "",
            "Profession-pere-mari": "",
            "Ville-residence-pere-mari": "",
            "Departement-residence-pere-mari": "",
            "Numero-residence-pere-mari": "",
            "Type-rue-residence-pere-mari": "",
            "Nom-rue-residence-pere-mari": "",
            "Prenom-mere-mari": "Jeanne Marie",
            "Nom-mere-mari": "POCARD-",
            "Profession-mere-mari": "",
            "Ville-residence-mere-mari": "",
            "Departement-residence-mere-mari": "",
            "Pays-residence-mere-mari": "",
            "Numero-rue-residence-mere-mari": "",
            "Type-rue-residence-mere-mari": "",
            "Nom-rue-residence-mere-mari": "",
            "Prenom-mariee": "Adrienne Jeanne",
            "Nom-mariee": "ALBRIEUX",
            "Profession-mariee": "journalière",
            "Ville-naissance-mariee": "Paris, onzième\narrondissement",
            "Departement-naissance-mariee": "",
            "Pays-naissance-mariee": "",
            "Jour-naissance-mariee": "onze",
            "Mois-naissance-mariee": "septembre",
            "Annee-naissance-mariee": "mil neuf cent deux",
            "Age-mariee": "trente-sept ans",
            "Ville-residence-mariee": "Paris",
            "Departement-residence-mariee": "",
            "Pays-residence-mariee": "",
            "Numero-rue-residence-mariee": "56",
            "Type-rue-residence-mariee": "rue",
            "Nom-rue-residence-mariee": "Saint Honoré",
            "Prenom-pere-mariee": "Marie Charles",
            "Nom-pere-mariee": "ALBRIEUX",
            "Profession-pere-mariee": "sans profession",
            "Ville-residence-pere-mariee": "Aulnay sous Bois",
            "Departement-residence-pere-mariee": "Sein \net Oise",
            "Numero-residence-pere-mariee": "",
            "Type-rue-residence-pere-mariee": "",
            "Nom-rue-residence-pere-mariee": "",
            "Prenom-mere-mariee": "Marguerite",
            "Nom-mere-mariee": "TERLES",
            "Profession-mere-mariee": "",
            "Ville-residence-mere-mariee": "",
            "Departement-residence-mere-mariee": "",
            "Pays-residence-mere-mariee": "",
            "Numero-rue-residence-mere-mariee": "",
            "Type-rue-residence-mere-mariee": "",
            "Nom-rue-residence-mere-mariee": "",
            "Prenom-de-l'ex-époux": "",
            "Nom-de-l'ex-époux": "",
            "Prenom-temoin-0": "Fernande",
            "Nom-temoin-0": "PASTEAU",
            "Profession-temoin-0": "concierge",
            "Age-temoin": "",
            "Numero-rue-residence-temoin-0": "13",
            "Type-rue-residence-temoin-0": "rue",
            "Nom-rue-residence-temoin-0": "Dussoubs",
            "Ville-residence-temoin": "",
            "Departement-residence-temoin": "",
            "Prenom-temoin-1": "Amélie",
            "Nom-temoin-1": "MASSONIE",
            "Profession-temoin-1": "ménagère",
            "Numero-rue-residence-temoin-1": "10",
            "Type-rue-residence-temoin-1": "rue",
            "Nom-rue-residence-temoin-1": "Volta"
        }
    
    distances = 0
    for key in ref.keys():
        distance = Levenshtein.distance(labels[key], ref[key])
        if distance > 0:
            print(key, distance, labels[key] if labels[key] != '' else 'VIDE', ref[key] if ref[key] != '' else 'VIDE')
        distances += distance

    print(distances)








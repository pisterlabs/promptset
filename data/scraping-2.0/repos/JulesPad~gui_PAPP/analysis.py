import sys
import json
from itertools import islice
import openai
import os

"""
def analyze_user_profile(first_name, last_name):
    # Add the analysis code here. For instance:
    with open(f"{first_name}_{last_name}_profile_ordered.json", 'r') as f:
        user_profile = json.load(f)

    """

INSTRUCTIONS = ""

def data_loading(first_name, last_name):

    with open(f"{first_name}_{last_name}_profile_ordered.json", 'r') as f:
        user_profile = json.load(f)

    # read the data from the JSON file
        data = user_profile



    aspiration = ("")

    # create the RIASEC dictionary
    RIASEC = {
        "REALISTE": data["REALISTE"],
        "INVESTIGATEUR": data["INVESTIGATEUR"],
        "ARTISTIQUE": data["ARTISTIQUE"],
        "SOCIAL": data["SOCIAL"],
        "ENTREPRENANT": data["ENTREPRENANT"],
        "CONVENTIONNEL": data["CONVENTIONNEL"]
    }

    # create the Categories dictionary
    Categories = {
        "PLEIN AIR & PHYSIQUE": data["PLEIN AIR & PHYSIQUE"],
        "PRATIQUE": data["PRATIQUE"],
        "TECHNIQUE": data["TECHNIQUE"],
        "SCIENTIFIQUE": data["SCIENTIFIQUE"],
        "COMMUNICATION": data["COMMUNICATION"],
        "ESTHETIQUE": data["ESTHETIQUE"],
        "SOUTIEN SOCIAL": data["SOUTIEN SOCIAL"],
        "SOINS MEDICAUX": data["SOINS MEDICAUX"],
        "NEGOCIATION": data["NEGOCIATION"],
        "LEADERSHIP": data["LEADERSHIP"],
        "TRAVAIL de BUREAU": data["TRAVAIL de BUREAU"],
        "INTÉRÊT POUR LES DONNÉES": data["INTÉRÊT POUR LES DONNÉES"]
    }

    # create the Domaines dictionary
    Domaines = {
        "AGRICULTURE et PECHE": data["AGRICULTURE et PECHE"],
        "MONDE ANIMALIER": data["MONDE ANIMALIER"],
        "SPORT": data["SPORT"],
        "FORCES DE L’ORDRE": data["FORCES DE L’ORDRE"],
        "BATIMENT ET TRAVAUX PUBLICS": data["BATIMENT ET TRAVAUX PUBLICS"],
        "TRANSPORTS": data["TRANSPORTS"],
        "HOTELLERIE, RESTAURATION et TOURISME": data["HOTELLERIE, RESTAURATION et TOURISME"],
        "MECANIQUE": data["MECANIQUE"],
        "DOMAINE INDUSTRIEL": data["DOMAINE INDUSTRIEL"],
        "ÉLECTRICITÉ": data["ÉLECTRICITÉ"],
        "SCIENCES DE LA TERRE et DE LA MATIERE": data["SCIENCES DE LA TERRE et DE LA MATIERE"],
        "SCIENCES DE LA VIE": data["SCIENCES DE LA VIE"],
        "MATHEMATIQUES": data["MATHEMATIQUES"],
        "ARTS DU SPECTACLE": data["ARTS DU SPECTACLE"],
        "LETTRES": data["LETTRES"],
        "ARTS GRAPHIQUES": data["ARTS GRAPHIQUES"],
        "ARTS APPLIQUES": data["ARTS APPLIQUES"],
        "ACCOMPAGNEMENT SOCIAL": data["ACCOMPAGNEMENT SOCIAL"],
        "ENSEIGNEMENT & FORMATION": data["ENSEIGNEMENT & FORMATION"],
        "PARAMEDICAL": data["PARAMEDICAL"],
        "VENTE DE PRODUITS ET DE SERVICES": data["VENTE DE PRODUITS ET DE SERVICES"],
        "MARKETING et PUBLICITE": data["MARKETING et PUBLICITE"],
        "JURIDIQUE et POLITIQUE": data["JURIDIQUE et POLITIQUE"],
        "MANAGEMENT": data["MANAGEMENT"],
        "RESSOURCES HUMAINES": data["RESSOURCES HUMAINES"],
        "ADMINISTRATION": data["ADMINISTRATION"],
        "COMPTABILITÉ et FINANCES": data["COMPTABILITÉ et FINANCES"],
        "INFORMATIQUE": data["INFORMATIQUE"]
    }

    sorted_domaines = sorted(Domaines.keys(), key=Domaines.get, reverse=True)

    return RIASEC, Categories, Domaines, user_profile, aspiration, sorted_domaines


def prompts():

    #definir le master prompt 

    masterPrompt = ("Tu es ChatGPT qui joue le rôle d'un psychologue spécialiste de l'orientation scolaire et professionnelle, surentraîner à analyser et à interpréter des tests à partir du modèle RIASEC de Holland. Tu travail à partir des résultats d'un test d'orientation qui propose trois niveaux de réponses. 1) Correspond aux TYPES (RAISEC) 2) Aux CATEGORIES (subdivision des TYPES) 3) Aux DOMAINES (subdivision des categories). Tu divises ton analyse en 3 parties : 1- Tu analyses les résultats bruts et examines les éléments avec les scores les plus élevés. 2- Tu analyses ce que la prédominance de ces éléments peut signifier. 3- Tu analyses l'implication générale pour le bénéficiaire du test.")

    #definir les prompts d'analyse 

    #prompt CATEGORIES
    categoriesPrompt = ("Analyse les résultats des catégories du bénéficiaire en trois parties comme mentionné précédemment. Pour les rèsultats bruts, prend seulement les 3 categories au score les plus élevés")

    #prompt Domaines 
    domainesPrompt = ("Ignore les intructions précedante. Voici les domaines déduis des resultats du bénéficiare. Analyse ces rèultats en 3 parties comme mentioné précedament. Les résulats suivant sont déjà classé par odre décroissant prend seulement les 6 domaines premier domaines. Apporte une conclusion précise et nuancé de l'implication des domaines trouvé pour le bénéficiare ")

    #prompt combination Domaine Profil type
    combinationPrompt =("À partir du profil stéréotypés, utilise les domaines afin de préciser les secteurs professionnels dans lesquels devrait travailler le bénéficiaire. Dans un second temps, en utilisant le profil de personalité stéreotypé conserve seulement les secteurs d'activité peritent.")

    #prompt Profil type
    profileType = ("Ignore les intructions précedante.Crée un profil stéréotypés des secteurs d'activité dans lesquelles le bénéficiaire pourrait travailler. Réalise ce profil à partir des catégories et des aspirations. Ajoute aussi un profil stéréotypés de sa personnalité de travailleurs. Ne donne pas de metiers")

    #prompt Metier

    metierPrompt = ("Ignore les intructions précedante. Propose une liste de métiers pertiants pour le béneficiare au regard de l'analyse suivante.")

    #prompt Coherence
    promptCoherence = ("")

    #prompt analyse rèsumé
    promptResume =("Ignore les intructions précedante. Rèsume cette analyse des rèsulats en conservant les points les plus importants et significatif de la 3éme partie SEULEMENT")

    return masterPrompt, categoriesPrompt, domainesPrompt, promptCoherence, promptResume, profileType, metierPrompt, combinationPrompt
def aspiration_engine(aspiration, RIASEC):
    
    print(type(RIASEC))
    # Sort the first 6 keys in descending order based on their values
    sorted_keys = sorted(RIASEC.keys(), key=RIASEC.get, reverse=True)
    # Take the first two keys with the highest values
    top_keys = sorted_keys[:2]


    if (top_keys[0] == 'REALISTE' and top_keys[1] == 'INVESTIGATEUR') or  (top_keys[0] == 'INVESTIGATEUR' and top_keys[1] == 'REALISTE'):
        aspiration = ("l'aspiration du bénéficiare est l'EXPERIMENTATION")
    elif (top_keys[0] == 'INVESTIGATEUR' and top_keys[1] == 'ARTISTIQUE') or  (top_keys[0] == 'ARTISTIQUE' and top_keys[1] == 'INVESTIGATEUR'):
        aspiration = ("l'aspiration du bénéficiare est la CONCEPTION")
    elif (top_keys[0] == 'ARTISTIQUE' and top_keys[1] == 'SOCIAL') or  (top_keys[0] == 'SOCIAL' and top_keys[1] == 'ARTISTIQUE'):
        aspiration = ("l'aspiration du bénéficiare est COMMUNICATION")
    elif (top_keys[0] == 'SOCIAL' and top_keys[1] == 'ENTREPRENANT') or  (top_keys[0] == 'ENTREPRENANT' and top_keys[1] == 'SOCIAL'):
        aspiration = ("l'aspiration du bénéficiare est RELATIONNEL")
    elif (top_keys[0] == 'ENTREPRENANT' and top_keys[1] == 'CONVENTIONNEL') or  (top_keys[0] == 'CONVENTIONNEL' and top_keys[1] == 'ENTREPRENANT'):
        aspiration = ("l'aspiration du bénéficiare est l'ORGANISATION")
    elif (top_keys[0] == 'CONVENTIONNEL' and top_keys[1] == 'REALISTE') or  (top_keys[0] == 'REALISTE' and top_keys[1] == 'CONVENTIONNEL'):
        aspiration = ("l'aspiration du bénéficiare est PROCEDURE")
    elif (top_keys[0] == 'REALISTE' and top_keys[1] == 'ARTISTIQUE') or  (top_keys[0] == 'ARTISTIQUE' and top_keys[1] == 'REALISTE'):
        aspiration = ("l'aspiration du bénéficiare est GOUT POUR LA REFLEXION INDIVIDUELLE")
    elif (top_keys[0] == 'REALISTE' and top_keys[1] == 'ENTREPRENANT') or  (top_keys[0] == 'ENTREPRENANT' and top_keys[1] == 'REALISTE'):
        aspiration = ("l'aspiration du bénéficiare est BESOIN D’ACTION")
    elif (top_keys[0] == 'INVESTIGATEUR' and top_keys[1] == 'SOCIAL') or  (top_keys[0] == 'SOCIAL' and top_keys[1] == 'INVESTIGATEUR'):
        aspiration = ("l'aspiration du bénéficiare est GOUT POUR LA RESOLUTION DE PROBLEME")
    elif (top_keys[0] == 'INVESTIGATEUR' and top_keys[1] == 'CONVENTIONNEL') or  (top_keys[0] == 'CONVENTIONNEL' and top_keys[1] == 'INVESTIGATEUR'):
        aspiration = ("l'aspiration du bénéficiare est GOUT POUR LE TRAVAIL SUR LES DONNEES")
    elif (top_keys[0] == 'ARTISTIQUE' and top_keys[1] == 'ENTREPRENANT') or  (top_keys[0] == 'ENTREPRENANT' and top_keys[1] == 'ARTISTIQUE'):
        aspiration = ("l'aspiration du bénéficiare est BESOIN D’AUTONOMIE DANS LE TRAVAIL")
    elif (top_keys[0] == 'SOCIAL' and top_keys[1] == 'CONVENTIONNEL') or  (top_keys[0] == 'CONVENTIONNEL' and top_keys[1] == 'SOCIAL'):
        aspiration = ("l'aspiration du bénéficiare est SENS DES RESPONSABILITES")
    elif (top_keys[0] == 'SOCIAL' and top_keys[1] == 'REALISTE') or  (top_keys[0] == 'REALISTE' and top_keys[1] == 'SOCIAL'):
        aspiration = ("l'aspiration du bénéficiare est GOUT POUR LES RESULTATS CONCRETS")
    elif (top_keys[0] == 'ARTISTIQUE' and top_keys[1] == 'CONVENTIONNEL') or  (top_keys[0] == 'CONVENTIONNEL' and top_keys[1] == 'ARTISTIQUE'):
        aspiration = ("l'aspiration du bénéficiare est SENS DU DETAIL")
    elif (top_keys[0] == 'INVESTIGATEUR' and top_keys[1] == 'ENTREPRENANT') or  (top_keys[0] == 'ENTREPRENANT' and top_keys[1] == 'INVESTIGATEUR'):
        aspiration = ("l'aspiration du bénéficiare est OUVERTURE D’ESPRIT")
    
    

    
    return aspiration

def chatGPT(masterPrompt, aspiration): 
    openai.api_key = "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"

    INSTRUCTIONS = masterPrompt + "" 

    chatMemory =""

    # on mettera a jour les intruction à chaque analyse en faissant ex: INSTRUCTIONS += categoriesprompt + chatMemory

    #pour ajouter des informations au fur à meusure faire INSTRUCTIONS += "intruction"

    TEMPERATURE = 0.4
    MAX_TOKENS = 800
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0
    
    return INSTRUCTIONS, TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY 


def get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, instructions):

    
    """Get a response from ChatCompletion
    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot
    Returns:
        The response text
    """
    # delteted the messages as we will be using our own memory module
    messages = [
        { "role": "system", "content": instructions },
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def rebootPrompt(masterPrompt, aspiration):
    INSTRUCTIONS = masterPrompt + "" + aspiration

    print("in the loop" + INSTRUCTIONS) 

    return INSTRUCTIONS





def main():
    
    # Load data
    RIASEC, Categories, Domaines, user_profile, aspiration, sorted_domaine = data_loading(first_name, last_name)

    # Get the prompts
    combinationPrompt, masterPrompt, categoriesPrompt, domainesPrompt, promptCoherence, promptResume, profileType, metierPrompt = prompts()
    

    # Get aspiration
    aspiration = aspiration_engine(aspiration, RIASEC)

    print(sorted_domaine)

    # Initial chatGPT setup
    INSTRUCTIONS, TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY = chatGPT(masterPrompt, aspiration)



    # -----------------------------------------------------  Analyse catègorie 
    INSTRUCTIONS += categoriesPrompt + str(Categories)

    print("DEBUG" + INSTRUCTIONS)

    analyse_detaille= get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    
    print("analyse detaillé categorie /n", analyse_detaille)

    # Reboot prompt
    INSTRUCTIONS = rebootPrompt(masterPrompt, aspiration)

    print("DEBUG" + INSTRUCTIONS)

    # Analyse resume
    INSTRUCTIONS += analyse_detaille + "" + promptResume 
    
    analyse_resume = get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    print("analyse résumé categorie /n",analyse_resume)


    # -----------------------------------------------------  Analyse profile type 
    # Reboot prompt
    INSTRUCTIONS = rebootPrompt(masterPrompt, aspiration)
    
    INSTRUCTIONS += profileType + " " + aspiration + " " + analyse_resume

    profileResulats = get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    
    print("profile type /n", profileResulats)


    # -----------------------------------------------------  Analyse Domaines 
    # Reboot prompt
    INSTRUCTIONS = rebootPrompt(masterPrompt, aspiration)
    
    #PENSEER A modifier le texte avant le profil type afin d'expliquer le profil type et specifier son utilisation

    INSTRUCTIONS += domainesPrompt + str(sorted_domaine)  
    analyse_detaille = get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    print("analyse domaine detaiullé/n", analyse_detaille)

    # Reboot prompt
    INSTRUCTIONS = rebootPrompt(masterPrompt, aspiration)

    # Analyse resume
    INSTRUCTIONS += analyse_detaille + "" + promptResume 
    
    analyse_resume = get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    print("analyse  domaine résumé/n",analyse_resume)

    # Reboot prompt
    INSTRUCTIONS = rebootPrompt(masterPrompt, aspiration)

    # Combination Domaine Porfil type 
    INSTRUCTIONS += combinationPrompt + " " + analyse_resume + " " + profileType 
    analyse_resume = get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    print("print canalyse combination /n", analyse_resume)


    # -----------------------------------------------------  Sugestion metier
    # Reboot prompt
    INSTRUCTIONS = rebootPrompt(masterPrompt, aspiration)
    
    INSTRUCTIONS += metierPrompt + " " + analyse_resume 

    recomendation_finale = get_analyse(TEMPERATURE, MAX_TOKENS, FREQUENCY_PENALTY, PRESENCE_PENALTY, INSTRUCTIONS)
    
    print("RECOMENDATION METIER /n", recomendation_finale)

    # ----------------------------------------------------- Affichage final

    with open(f"analysis.txt", 'w') as f:
        f.write(analyse_resume)
        f.flush()


if __name__ == "__main__":
    first_name = sys.argv[1]
    last_name = sys.argv[2]
    data_loading(first_name=first_name, last_name=last_name)
    main()

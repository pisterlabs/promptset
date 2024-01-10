import openai

#Remplacez par votre clé API OpenAI
api_key = "VOTRE_CLE_API_OPENAI"

def generer_deck(difficulte, historique):
    openai.api_key = api_key
    prompt = f"Générer un deck de solitaire de difficulté '{difficulte}' en fonction de l'historique : {historique}"

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100  # Ajustez la longueur de la réponse au besoin
    )
    deck = response.choices[0].text.strip()
    return deck

def jouer_solitaire(difficulte, historique):
    deck = generer_deck(difficulte, historique)

#Utilisez le deck généré pour jouer au solitaire
    print(f"Deck de solitaire généré : {deck}")

if name == "main":
    difficulte = "difficile"  # Vous pouvez obtenir la difficulté de l'utilisateur
    historique = "Historique des parties précédentes"  # Obtenez l'historique des parties de l'utilisateur
    jouer_solitaire(difficulte, historique)
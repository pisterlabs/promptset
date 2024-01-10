import os
import openai
from dotenv import load_dotenv

# chargement des variables d'environnement à partir du fichier .env
load_dotenv()

# Définition de la clé d'API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialise un dictionnaire qui va garder l'historique de la conversation pour chaque session
# Chaque clé dans le dictionnaire représente un identifiant de session et la valeur associée est une liste des messages échangés durant la session
chat_histories = {}
print("chat_histories", chat_histories)

def chat_service(model, data):

    # Récupération de l'identifiant de session (session_id) et le message de l'utilisateur (query) à partir des données reçues
    # session_id est utilisé pour associer un historique de chat à chaque session d'utilisateur
    session_id = data.get("session_id")
    query = data.get("query")

    # Vérification si l'historique de chat existe déjà pour la session_id. Si non cela renvoie une liste vide
    chat_history = chat_histories.get(session_id, [])
    print('chat_history :', chat_history)

    # Ajout du nouveau message à l'historique de chat
    chat_history.append((query, ""))
    print('chat_history avant append :', chat_history)

    # Construction de la liste des messages à envoyer à l'API d'OpenAI pour obtenir une réponse
    # Chaque message est stocké dans un dictionnaire
    messages = [{"role": "user", "content": msg} for msg, _ in chat_history]

    # Appel à l'API d'OpenAI pour obtenir la réponse
    # Utilisation du paramètre model en fonction du choix du LLM par l'utilisateur côté fontend
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages)

    # Extraction de la réponse de l'API
    assistant_reply = response["choices"][0]["message"]["content"]

    # Mise à jour du dernier élément de l'historique de chat avec la réponse de l'assistant
    chat_history[-1] = (query, assistant_reply)
    print('chat_history[-1] :', chat_history[-1])

    # Mise à jour de l'historique de chat pour la session de l'utilisateur dans le dictionnaire chat_histories
    chat_histories[session_id] = chat_history
    print('chat_histories[session_id] :', chat_histories[session_id])

    # Renvoie de la réponse de l'assistant
    return {"answer": assistant_reply}
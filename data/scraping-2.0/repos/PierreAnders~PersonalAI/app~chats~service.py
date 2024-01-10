import os
import sys
from flask_jwt_extended import get_jwt_identity
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from glob import glob
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import datetime
from app.users.model import User

# chargement des variables d'environnement à partir du fichier .env
load_dotenv()

# Définition de la clé d'API de OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialise un dictionnaire qui va garder l'historique de la conversation pour chaque session
chat_histories = {}
print("chat_histories", chat_histories)


def write_date(user_id):
    user_subfolder_info_db = os.path.join('data', str(user_id), f"info-{user_id}")
    try:
        os.makedirs(user_subfolder_info_db, exist_ok=True)
        print(f"Dossier '{user_subfolder_info_db}' créé avec succès.")
    except FileExistsError:
        print(f"Le dossier '{user_subfolder_info_db}' existe déjà.")
    except Exception as e:
        print(f"Une erreur s'est produite lors de la création du dossier : {str(e)}")
    file_path = os.path.join(user_subfolder_info_db, 'date.txt')
    print(datetime.date.today())
    current_date = datetime.date.today().strftime('%Y-%m-%d')
    print(current_date)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"INFORMATIONS DU JOUR:\n\n")
        file.write(f"Nous sommes aujourd'hui le {current_date}")
        

def chat_with_data_service(model, data):

    user_id = get_jwt_identity()
    write_date(user_id)

    # Récupération ou création de l'index
    index = get_or_create_index()
    print('index :', index)

    # Création la chaîne de conversation
    chain = create_conversational_chain(model, index)
    print('chain :', chain)

    # Récupération l'ID de la session et de la requête de l'utilisateur
    session_id = data.get("session_id")
    print('session_id :', session_id)
    query = data.get("query")
    print('query :', query)

    # Récupération de l'historique de la session
    chat_history = get_chat_history(session_id, chat_histories)
    print('chat_history :', chat_history)

    # Génération de la réponse à la requête
    result = generate_response(chain, query, chat_history)
    print('result :', result)

    # Mise à jour de l'historique de la conversation
    update_chat_history(session_id, chat_history, query, result['answer'])
    print('chat_history :', chat_history)

    return result


# Récupération ou création de l'index
def get_or_create_index():
    # Création d'un nouvel index à partir des données du dossier de l'utilisateur
    user_id = get_jwt_identity()
    user_data_folder = f'data/{user_id}/'
    # "os.scandir(user_data_folder)" parcourt tous les éléments dans le répertoire "user_data_folder"
    # "if folder.is_dir()" vérifie si l'élément actuellement observé "folder" est un répertoire.
    # Si "folder" est un répertoire, alors son chemin d'accès est obtenu avec "folder.path".
    subdirs = [folder.path for folder in os.scandir(user_data_folder) if folder.is_dir()]
    print('subdirs :', subdirs)
    # Création d'un chargeur de documents pour chaque sous-répertoire
    loaders = [DirectoryLoader(subdir) for subdir in subdirs]
    print ('loaders :', loaders)
    # Création de l'index
    index = VectorstoreIndexCreator().from_loaders(loaders)
    print('index :', index)

    return index


# Création d'une instance de la classe ConversationalRetrievalChain  
# pour gérer les conversations et récupérer des informations pertinentes 
# en fonction du modèle de langage et du récupérateur fournis.
# llm crée une instance de la classe ChatOpenAI avec comme modèle gpt-4
# Retriever représente le composant récupérateur de la chaîne, 
# responsable de la récupération d'informations pertinentes
# en appelant index.vectorstore.as_retriever() pour convertir 
# le stockage de vecteurs en un récupérateur.
# search_kwarks indique que seul un résultat doit être extrait.
def create_conversational_chain(model, index):
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model=model),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )
    print('chain :', chain)
    return chain


# Récupération de l'historique de chat de la session
def get_chat_history(session_id, chat_histories):
    chat_history = chat_histories.get(session_id, [])
    print('chat_history :', chat_history)
    return chat_history


# Génération de la réponse à la requête
def generate_response(chain, query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    print('result :', result)
    return result


# Mise à jour de l'historique de conversation
def update_chat_history(session_id, chat_history, query, answer):
    # "chat_histories" est un dictionnaire dont les clés sont les identifiants de session "session_id" 
    # et les valeurs sont les historiques de chat pour chaque session.
    # "chat_histories[session_id]" accède à l'historique de chat pour la session actuelle. 
    #  Si aucune session avec cet ID n'existe encore, une nouvelle entrée sera créée.
    # "= chat_history" assigne l'historique de chat mis à jour à cette entrée dans le dictionnaire. 
    chat_history.append((query, answer))
    print('chat_history :', chat_history)
    chat_histories[session_id] = chat_history
    print('chat_histories[session_id] :', chat_histories[session_id])


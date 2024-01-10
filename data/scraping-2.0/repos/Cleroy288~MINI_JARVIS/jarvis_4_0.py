import requests
import speech_recognition as sr
import pyttsx3
import openai
import os
import sys
import time
import re
import shutil

from bs4 import BeautifulSoup
from cryptography.fernet import Fernet
from colorama import Fore, init
from PIL import Image
from io import BytesIO

# Constants
WEATHER_API_BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
MODEL_NAME = "gpt-3.5-turbo"

# Initialisation
init(autoreset=True)

def print_slow(text, color):
    for char in text:
        sys.stdout.write(color + char)
        sys.stdout.flush()
        time.sleep(0.025)
    print('')

def colored_print(text, color):
    print(color + text)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def search_and_get_info(query):
    GOOGLE_CSE_API_KEY = keys["google_cse"]
    GOOGLE_CSE_CX = keys["google_cse_cx"]
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_CX}"

    response = requests.get(search_url)
    result = response.json()

    if result['searchInformation']['totalResults'] == '0':
        return "Désolé, je n'ai pas pu trouver d'informations pertinentes."

    top_result_link = result['items'][0]['link']
    page_content = requests.get(top_result_link).text
    soup = BeautifulSoup(page_content, 'html.parser')
    paragraphs = soup.find_all('p')

    # Combinez les 3 premiers paragraphes pour une réponse plus détaillée.
    combined_text = ' '.join([p.get_text() for p in paragraphs[:3]])

    if combined_text:
        return combined_text
    else:
        return f"Je n'ai pas pu extraire d'informations pertinentes de la page, mais vous pouvez la consulter directement ici : {top_result_link}"

def extract_code_from_response(text):
    # Utilisez des expressions régulières pour trouver du code entre triple backticks
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

##############################################################################################
def generate_code_with_context(prompt, filename=None, existing_code=None):
    # Détecte le langage désiré à partir du nom du fichier
    file_extension = filename.split('.')[-1] if filename else None
    lang = "Python"
    if file_extension == "c":
        lang = "C"
    elif file_extension == "js":
        lang = "JavaScript"
    elif file_extension == "java":
        lang = "Java"
    # ... ajoutez d'autres extensions et leurs langages respectifs au besoin

    # Modifiez le message du système en fonction du langage détecté
    system_msg = f"Your instruction is to generate {lang} code based on the given prompt."

    if existing_code:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": existing_code},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ]

    # Appel à l'API ChatGPT
    response = openai.ChatCompletion.create(
      model=MODEL_NAME,
      messages=messages
    )

    ai_response = response.choices[0].message['content']

    # Extrait le code de la réponse
    generated_code = extract_code_from_response(ai_response)

    # Filtrer le nom du langage de la première ligne s'il existe
    if generated_code:
        first_line = generated_code.split('\n')[0].strip().lower()
        known_languages = ['python', 'java', 'javascript', 'c']
        if first_line in known_languages:
            generated_code = '\n'.join(generated_code.split('\n')[1:]).strip()

    if not generated_code:
        return "Je n'ai pas pu générer le code comme demandé."

    # Écrire le code dans un fichier
    if filename:
        with open(filename, 'w') as file:
            file.write(generated_code)

    return generated_code

##################################################################################################

def improve_code(file_path, prompt):
    # Vérifie si le fichier existe
    if not os.path.exists(file_path):
        print(f"Le fichier {file_path} n'existe pas.")
        return

    # Lire le contenu du fichier
    with open(file_path, 'r') as file:
        code_content = file.read()

    # Préparer le dossier pour les anciens fichiers
    old_files_path = 'old_files'
    specific_old_file_path = os.path.join(old_files_path, os.path.basename(file_path).split('.')[0])
    if not os.path.exists(old_files_path):
        os.mkdir(old_files_path)
    if not os.path.exists(specific_old_file_path):
        os.mkdir(specific_old_file_path)

    # Copier le fichier original dans le dossier des anciens fichiers
    index = len(os.listdir(specific_old_file_path)) + 1
    shutil.copy(file_path, os.path.join(specific_old_file_path, f"{os.path.basename(file_path).split('.')[0]}_{index}.py"))

    # Appeler OpenAI pour obtenir des suggestions
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": code_content},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="text-davinci-002",
        messages=messages
    )

    ai_response = response.choices[0].message['content']

    # Vérifiez si la réponse est une amélioration du code ou non
    if ai_response and ai_response != code_content:
        # Écrire le nouveau code dans le fichier original
        with open(file_path, 'w') as file:
            file.write(ai_response)
        print("Améliorations terminées !")
    else:
        print("Pas d'améliorations suggérées pour le code.")

    # Enregistrez également le prompt pour la traçabilité
    with open(os.path.join(specific_old_file_path, f"prompt_{index}.txt"), 'w') as prompt_file:
        prompt_file.write(prompt)


###########################################################################

def is_valid_filename(filename):
    """Vérifie si le nom de fichier est valide"""
    return bool(re.match(r'^[\w\-. ]+$', filename))

def show_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.show()

def get_weather(city, api_key):
    complete_url = WEATHER_API_BASE_URL + "appid=" + api_key + "&q=" + city
    response = requests.get(complete_url)
    data = response.json()
    if data["cod"] != "404":
        main_data = data["main"]
        weather_data = data["weather"][0]
        temperature = main_data["temp"] - 273.15
        weather_description = weather_data["description"]
        return f"La température à {city} est de {temperature:.2f}°C avec {weather_description}."
    else:
        return f"Je n'ai pas pu trouver les données météorologiques pour {city}."

def decrypt_keys(crypto_key):
    cipher_suite = Fernet(crypto_key.encode())
    with open("encrypted_keys.txt", "rb") as f:
        lines = f.readlines()
    decrypted_keys = {}
    for line in lines:
        api_name, encrypted_key = line.decode().split(" ==> ")
        decrypted_keys[api_name] = cipher_suite.decrypt(encrypted_key.strip().encode()).decode()
    return decrypted_keys

def system_speak_text(text):
    escaped_text = text.replace('"', '\\"')
    os.system(f'say "{escaped_text}"')

# Loading keys and setup
with open("keys/crypto_key.key", "rb") as f:
    CRYPTO_KEY = f.read().decode()

keys = decrypt_keys(CRYPTO_KEY)
openai.api_key = keys["chat gpt"]

r = sr.Recognizer()
mic = sr.Microphone()

conversation = [
    {"role": "system", "content": "Your name is Jarvis and your purpose is to be Charles AI assistant, always respond quickly and in french"},
]


try:
    while True:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            colored_print("System :: [parlez maintenant]", Fore.GREEN)
            
            try:
                audio = r.listen(source)
                word = r.recognize_google(audio, language='fr-FR')
                colored_print(f"UTILISATEUR :: [{word}]", Fore.YELLOW)

                # Pour la météo
                if "météo" in word.lower() or "température" in word.lower():
                    city = word.split("à")[-1].strip()
                    weather_info = get_weather(city, keys["openweather"])
                    colored_print(weather_info, Fore.BLUE)
                    system_speak_text(weather_info)

                # Pour dessiner une image
                elif "dessine-moi" in word or "peux-tu dessiner" in word:
                    prompt = word
                    response = openai.Completion.create(
                      model=MODEL_NAME,
                      prompt=f"({prompt}) 'S'il te plaît dessine-moi une image en réponse.'",
                      max_tokens=150
                    )
                    image_url = response.choices[0].text.strip()
                    show_image_from_url(image_url)

                # Pour générer du code
                elif "code" in word or "script" in word or "programme" in word:
                    prompt = word
                    filename = input("Entrez le nom de votre fichier (ou laissez vide pour afficher à l'écran) : ").strip()
                    if filename and not is_valid_filename(filename):
                        colored_print("Le nom de fichier fourni n'est pas valide. Le code sera affiché à l'écran.", Fore.RED)
                        filename = None
                    code = generate_code_with_context(prompt, filename=filename)
                    if not filename:
                        colored_print(code, Fore.BLUE)
                    else:
                        colored_print(f"Le code a été écrit dans {filename}", Fore.BLUE)

                # Pour améliorer/optimiser/ vérifier du code
                elif any(keyword in word for keyword in ["améliorer", "optimiser", "modifier", "vérifier", "checker"]):
                    filename = input("Entrez le nom du fichier à améliorer (ou laissez vide pour annuler) : ").strip()
                    if filename:
                        if os.path.exists(filename):
                            improve_code(filename, word)
                            colored_print(f"Le fichier {filename} a été amélioré.", Fore.BLUE)
                        else:
                            colored_print(f"Le fichier {filename} n'existe pas. Assurez-vous d'entrer le bon chemin.", Fore.RED)
                    else:
                        colored_print("Action annulée.", Fore.BLUE)

                # Pour rechercher des informations
                elif "recherche" in word or "cherche-moi" in word or "informations" in word:
                    search_info = search_and_get_info(word.split("sur")[-1].strip())
                    colored_print(search_info, Fore.BLUE)
                    system_speak_text(search_info)

                # Terminer le programme
                elif "c'est la fin" in word.lower() or "au revoir" in word.lower():
                    goodbye_message = "Au revoir ! C'était un plaisir de vous assister."
                    colored_print(goodbye_message, Fore.BLUE)
                    system_speak_text(goodbye_message)
                    break

                # Autres interactions
                else:
                    conversation.append({
                        "role": "user",
                        "content": word
                    })
                    response = openai.ChatCompletion.create(
                      model=MODEL_NAME,
                      messages=conversation
                    )
                    response_text = response.choices[0].message['content']
                    colored_print(f"IA :: [{response_text}]", Fore.BLUE)
                    system_speak_text(response_text)
                    conversation.append({
                        "role": "assistant",
                        "content": response_text
                    })

            except sr.UnknownValueError:
                colored_print("Désolé, je n'ai pas compris ce que vous avez dit.", Fore.RED)
                system_speak_text("Désolé, je n'ai pas compris ce que vous avez dit.")
            except sr.RequestError:
                colored_print("Il y a un problème de connexion au service de reconnaissance vocale.", Fore.RED)
                system_speak_text("Il y a un problème de connexion au service de reconnaissance vocale.")
            except Exception as e:
                colored_print(f"Erreur: {e}", Fore.RED)

except KeyboardInterrupt:
    colored_print("Arrêt du programme. À bientôt!", Fore.GREEN)
    system_speak_text("Arrêt du programme. À bientôt!")

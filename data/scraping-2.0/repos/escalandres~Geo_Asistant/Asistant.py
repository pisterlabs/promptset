gptToken = "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"
#1
import openai
# Configurar la API de OpenAI
openai.api_key = gptToken
import wolframalpha
app_id = "RGKTU7-VYTGGTHK2Y"
client = wolframalpha.Client(app_id)
import speech_recognition as sr
import pyttsx3
import pywhatkit
import wikipediaapi
import wikipedia
import os
from gtts import gTTS
import requests
import json
#pip install requests
#-----------------------------------------------------------------------------------------
#Declaracion de funciones
def talk(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

def take_command():
    command = ""
    try:
        with sr.Microphone() as source:
            #print("Adjusting for background noise. One second")
            listener.adjust_for_ambient_noise(source)
            #talk("Ok, I'm listening to you")
            print('Te escucho...')
            voice = listener.listen(source, phrase_time_limit=10)
            command = listener.recognize_google(voice, language="es-ES")
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '')
            #print(command)
    except LookupError:   # speech is unintelligible
        print("Could not understand audio")
        pass
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:
        print("Error:", e)
    except:
        print('Goodbye')
    return command

def get_info():
    try:
        info = ""
        with sr.Microphone() as source:
            listener.adjust_for_ambient_noise(source)
            talk('¿Cuál mineral quieres buscar?')
            voice = listener.listen(source,phrase_time_limit=10)
            info = listener.recognize_google(voice, language="es-ES")
            info = info.lower()
    except LookupError:   # speech is unintelligible
        print("Could not understand audio")
        pass
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    except Exception as e:
        print("Error:", e)
    except:
        pass
    return info

def obtener_respuesta(pregunta):
    res = ""
    try:
        respuesta = openai.Completion.create(
            engine='text-davinci-003',  # Utiliza el motor 'text-davinci-003' para ChatGPT
            prompt=pregunta,
            max_tokens=100,  # Define la longitud máxima de la respuesta generada
            temperature=0.7,  # Controla la creatividad de la respuesta generada
            n=1,  # Especifica el número de respuestas a generar
            stop=None  # Puedes definir una cadena para detener la respuesta en un punto específico
        )
        res = respuesta.choices[0].text.strip()
    except Exception as e:
        print("\nOcurrió un error. Error: " + e)
    return res

def run_alexa():
    command = take_command()
    talk('Procesando tu petición...')
    if 'reproduce' in command:
        song = command.replace('reproduce', '')
        print('Reproduciendo: ' + song)
        talk('Reproduciendo: ' + song)
        pywhatkit.playonyt(song)
    elif 'cuales son' in command:
        try:
            print(command)
            person = command
            wolfram_res = next(client.query(person).results).text
            talk(wolfram_res)
        except Exception as e:
            print("Error:", e)
    elif 'muéstrame más información sobre' in command:
        person = command.replace('muéstrame más información sobre', '')
        info = wikipedia.summary(person, 1)
        print('wikipedia result: '+info)
        talk('wikipedia result: '+info)
    elif 'qué es' in command:
        search_term = command
        language = "es"
        wiki_wiki = wikipediaapi.Wikipedia(language)
        page = wiki_wiki.page(search_term)
        if page.exists():
            print("Título:", page.title)
            print("Contenido:", page.text)
        else:
            print("La página no existe.")

    elif 'dime más sobre los minerales' in command:
        pregunta_usuario = get_info()
        url = "https://es.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": pregunta_usuario + "(elemento)",
            "exintro": True,
            "explaintext": True
        }
        response = requests.get(url, params=params)
        data = response.json()
        page = data["query"]["pages"].popitem()[1]
        # sg.PopupNonBlocking(page["extract"])
        print(page["extract"])
        talk(page["extract"])

    elif 'minerales' in command:
        pregunta_usuario = get_info()
        # Crear un objeto de la API de Wikipedia
        wiki_wiki = wikipediaapi.Wikipedia('es')

        # Buscar un mineral y obtener su página
        mineral = input("Ingrese el nombre del mineral: ")
        page = wiki_wiki.page(mineral)

        if page.exists():
            # Obtener el contenido del resumen del mineral
            resumen = page.summary
            print("Resumen:")
            print(resumen)
        else:
            print("No se encontró información sobre el mineral.")

    elif 'busca' in command:
        #pregunta_usuario = get_info()
        pregunta_usuario = input("Hazme una pregunta: ")
        respuesta_chatgpt = obtener_respuesta(pregunta_usuario)
        talk(respuesta_chatgpt)
    elif 'mónica' in command:
        try:
            # url = "https://app.monicahq.com/api"
            # texto = "Muestrame todas las propiedades del mineral mercurio"
            # payload = {'text': texto}
            # headers = {'Authorization': 'Token ' + monicaToken}

            # response = requests.post(url, headers=headers, data=payload)
            # data = json.loads(response.text)
            # sentimientos = data['sentiments']
            # entidades = data['entities']
            # client = MonicaClient(access_token=monicaToken, api_url='https://app.monicahq.com/api')
            # print(client.me)
            print('j')
        except Exception as e:
            print("Error:", e)
    elif 'adiós' or 'apagate' or 'hasta luego' in command:
        global close
        close=1
        
    elif 'buscqa' in command:
        search = command.replace('busca', '')
        pywhatkit.search(search)
    
    else:
        talk('NO te entendí, ¿puedes repetirlo?')

#-----------------------------------------------------------------------------------------#
#Condigo principal
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
print(voices)
engine.setProperty('voice', voices[3].id)
close=0
os.system('cls' if os.name == 'nt' else 'clear')
talk('Hola, soy Alexa. ¿Cómo puedo ayudarte?')
while True:
    run_alexa()
    if close == 1:
        talk('¡Que tengas un buen día!')
        break
    talk('¿Necesitas algo más?...')
    
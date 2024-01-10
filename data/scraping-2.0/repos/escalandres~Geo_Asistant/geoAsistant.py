gptToken = "sk-38NsXfKe0dqqHd31Hc2wT3BlbkFJ3gHYTIU2q5ii4UlOfSe"
#1
from unidecode import unidecode
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
    print('Geo: '+text)
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
            voice = listener.listen(source, phrase_time_limit=7)
            command = listener.recognize_google(voice, language="es-ES")
            command = command.lower()
            if 'geo' in command:
                command = command.replace('geo', '')
            print(command)
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
    return unidecode(command.lower())

def get_info(message):
    try:
        info = ""
        with sr.Microphone() as source:
            listener.adjust_for_ambient_noise(source)
            talk(message)
            print('Geo: Te escucho...')
            voice = listener.listen(source,phrase_time_limit=7)
            info = listener.recognize_google(voice, language="es-ES")
            info = info.lower()
            print(info)
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

def run_geo():
    command = take_command()
    talk('Procesando tu petición...')
    if 'quiero informacion sobre una persona' in command:
        try:
            person = get_info('¿A quién quieres buscar?')
            wolfram_res = next(client.query(person).results).text
            talk(wolfram_res)
        except Exception as e:
            print("Error:", e)
    elif 'muestrame mas informacion' in command:
        person = get_info('¿Qué quieres saber?')
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

    elif 'quiero saber mas sobre los minerales' in command:
        pregunta_usuario = get_info('¿Qué quieres saber sobre los minerales?')
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

    elif 'yo tengo una duda' in command:
        #pregunta_usuario = get_info()
        # pregunta_usuario = input("Hazme una pregunta: ")
        pregunta_usuario = get_info('¿Qué quieres saber?')
        respuesta_chatgpt = obtener_respuesta(pregunta_usuario)
        talk(respuesta_chatgpt)
    elif 'adios' in command:
        global close
        close=1
    elif 'busca en google' in command:
        search = get_info('¿Qué quieres buscar en Google?')
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
talk('Hola, mi nombre es Geo. Soy un asistente virtual. ¿Cómo puedo ayudarte?')
while True:
    run_geo()
    if close == 1:
        talk('¡Que tengas un buen día!')
        break
    talk('¿Necesitas algo más?...')
    
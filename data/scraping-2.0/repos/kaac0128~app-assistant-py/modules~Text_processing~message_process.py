# Importar librerías
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
import requests
from modules.Tools.tools import *
from modules.Tools.search import *
from modules.Tools.Info import *
from modules.langchain_assitant.langchain_brain import LangChainBrainAssitant
from modules.You_opcions.you_chat import You_data

langChatBrain = LangChainBrainAssitant()


class TextProcessing:
    def createResponse(self, status=None, message=None, data=None):
        response = {"status": status, "data": data or {}, "message": message}
        return response

    def messageProcessClean(self, message):
        # Procesar el string
        tokens = nltk.word_tokenize(
            message.lower()
        )  # Tokenizar y convertir a minúsculas
        stop_words = stopwords.words(
            "spanish"
        )  # Definir las palabras vacías en español
        tokens = [
            token for token in tokens if token not in stop_words and token.isalpha()
        ]  # Eliminar las palabras vacías y los signos de puntuación
        lemmatizer = WordNetLemmatizer()  # Crear un lematizador
        tokens = [
            lemmatizer.lemmatize(token) for token in tokens
        ]  # Lematizar los tokens

        # Crear un diccionario y un corpus a partir de los tokens
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow([token]) for token in tokens]

        # Aplicar el modelo LDA con 1 tema
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=1, id2word=dictionary, passes=10
        )

        # Imprimir el tema principal del string
        return lda_model.print_topics()[0][1]

    def validationIA(self):
        url = "https://api.betterapi.net/youdotcom/chat"
        res = requests.get(url)
        if res:
            return True
        else:
            return False

    def selectCommand(self, command):
        commandClean = self.messageProcessClean(command)
        validationIA = self.validationIA()
        
        youService = You_data()
        command = command.lower()
        try:
            if "busc" in command or "consult" in command:
                if "google" in command:
                    response = self.createResponse(
                        "ok",
                        f"Aqui tienes una lista de los enlaces sobre {commandClean}",
                        search_google(commandClean),
                    )
                    return response
                elif "wiki" in command:
                    response = self.createResponse("ok", search_wikipedia(commandClean))
                    return response
                else:
                    if validationIA:
                        response = self.createResponse("ok", youService.chatYou(command))
                        return response
                    dataResponse = LangChainBrainAssitant.chat(command)
                    response = self.createResponse("ok", dataResponse["content"])
                    return response
            elif "abrir" in command:
                if "visual" in command:
                    response = self.createResponse("ok", openVsc()["message"])
                    return response
                elif "nautilus" in command:
                    response = self.createResponse("ok", openNautilus()["message"])
                    return response
                elif "navegador" in command or "chrome" in command:
                    url = "https://www.google.com"
                    response = self.createResponse("ok", openBrowser(url)["message"])
                    return response
                elif "notas" in command:
                    response = self.createResponse("ok", openGedit()["message"])
                    return response
                elif "youtube" in command:
                    url = "https://www.youtube.com/"
                    response = self.createResponse("ok", openBrowser(url)["message"])
                    return response
                elif "google" in command:
                    url = "https://www.google.com"
                    response = self.createResponse("ok", openBrowser(url)["message"])
                    return response
                else:
                    texto_modificado = command.replace(" ", "+")
                    url = "https://www.google.com/search?q=" + texto_modificado
                    response = self.createResponse("ok", openBrowser(url)["message"])
                    return response
            elif "hola" in command:
                response = self.createResponse(
                    "ok",
                    "Hola mi nombre es zero tu asistente virtual personal en que puedo ayudarte?",
                )
                return response
            elif "hora" in command:
                response = self.createResponse("ok", get_hour())
                return response
            elif "clima" in command:
                response = self.createResponse("ok", get_wheater())
                return response
            elif "direccion ip" in command:
                response = self.createResponse("ok", find_my_ip())
                return response
            elif "crea" in command or "agrega" in command:
                if "nota" in command:
                    response = self.createResponse(
                        "ok",
                        "Por favor escribe en el siguiente modal el titulo y el contenido de la nota",
                        "NOTIONDATA",
                    )
                else:
                    if validationIA:
                        response = self.createResponse("ok", youService.chatYou(command))
                        return response
                    dataResponse = LangChainBrainAssitant.chat(command)
                    response = self.createResponse("ok", dataResponse["content"])
                return response
            elif "envia" in command:
                if "correo" in command or "email" in command:
                    response = self.createResponse(
                        "ok",
                        "Por favor escribe el titulo, contenido y correo de lo que deseas enviar en el siguiente modal",
                        "EMAILDATA",
                    )
                    return response
                elif "whatsapp" in command:
                    response = self.createResponse(
                        "ok",
                        "Por favor escribe el numero de celular con el identificador de pais con el mensaje en el siguiente modal",
                        "WHATSAPPDATA",
                    )
                    return response
                else:
                    response = self.createResponse(
                        "ok", "no tengo la funcionalidad a donde lo quieres enviar"
                    )
                    return response
            elif "ver" in command or "reprodu" in command:
                if "video" in command or "musica" in command or "cancion" in command:
                    response = self.createResponse("ok", play_youtube(commandClean))
                    return response
                else:
                    response = self.createResponse(
                        "ok",
                        f"Aqui tienes una lista de los enlaces sobre {commandClean}",
                        search_google(commandClean),
                    )
                    return response
            elif "progra" in command or "agenda" in command:
                response = self.createResponse(
                    "ok",
                    "por favor en el siguiente modal ingresa los campos para agendar tu peticion",
                )
                return response
            else:
                if validationIA:
                    response = self.createResponse("ok", youService.chatYou(command))
                    return response
                dataResponse = LangChainBrainAssitant.chat(command)
                response = self.createResponse("ok", dataResponse["content"])
                return response
        except Exception as e:
            response = self.createResponse("error", "Error al procesar la solicitud", e)
            return response

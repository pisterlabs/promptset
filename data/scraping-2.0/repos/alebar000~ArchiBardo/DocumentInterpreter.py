import openai
import tiktoken

class DocumentInterpreter:

    def __init__(self):
        # Leer el key del OpenAI API desde el archivo APIKEY
        with open("APIKEY", "r") as file:
            apikey = file.read()
        openai.api_key = apikey

    def tokenCounter(self, text, chosenModel):
        # Contar todos los tokens del texto usando tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(chosenModel)
        tokenCount = len(enc.encode(text))

        return tokenCount
    
    def interpret(self, documentName, text, chosenModel, currentUnit, types):

        tryCounter = 0
        messages = [{"role": "system", "content": "Eres un sistema experto en realizar el proceso archivístico de clasificación documental. Vas a ayudarme a clasificar un documento."}]
        text = self.cleanText(text)
        newQuery = "Eres un sistema experto en realizar el proceso archivístico de clasificación documental. Tu objeto de análisis corresponde a un documento que ha sido enviado o recibido por la unidad organizacional llamada '" + currentUnit + "'. Responde únicamente lo solicitado, en formato CSV válido."
        newQuery += "El texto del documento es el siguiente: [[" + text + "]]."
        newQuery += "Según el texto anterior indicar la siguiente información, en formato CSV válido: 'año_del_documento, tipo_de_documento'." 
        newQuery += "El tipo_de_documento DEBE ser elegido de esta lista CSV de tipos: '" + types + "', y no debe ser ningún otro posible tipo. El año_del_documento DEBE ser un número entero, y lo obtienes del texto." 
        messages.append({"role": "user", "content": newQuery})
        newResponse = self.analyzeText(messages, chosenModel)
        messages.append({"role": "assistant", "content": newResponse})
        tryCounter += 1

        queryExitoso = False
        while not queryExitoso:
            print ("Intento número: ", tryCounter)
            # Separar la respuesta recibida en una lista de strings con el separador ','
            newResponseSplit = newResponse.split(",")
            # Eliminar todos los espacios al principio y final de cada string de la lista
            newResponseSplit = [x.strip() for x in newResponseSplit]
            print ("Respuesta: ", newResponseSplit)

            # Si ya son más de 3 intentos, asignar un valor por defecto a la respuesta
            if tryCounter > 3:
                newResponse = "2023, Correspondencia"
                queryExitoso = True
            # Si la respuesta no corresponde a una lista de dos elementos, generar una nueva query y analizarla
            elif len(newResponseSplit) != 2:
                newQuery = "Me diste una respuesta incorrecta, pues yo solo esperaba dos elementos separados por una coma. Responde nuevamente pero SOLAMENTE con el año_del_documento y el tipo_de_documento, en formato CSV válido."
                messages.append({"role": "user", "content": newQuery})
                newResponse = self.analyzeText(messages, chosenModel)
                messages.append({"role": "assistant", "content": newResponse})
                tryCounter += 1
            # Si el tipo indicado no se encuentra en la lista de tipos, generar una nueva query y analizarla
            elif newResponseSplit[1] not in types:
                newQuery = "Me diste una respuesta incorrecta, pues el tipo_de_documento que me indicaste no pertenece a la lista CSV de tipos que ya te indiqué. Responde nuevamente SOLAMENTE con el año_del_documento y el tipo_de_documento proveniente de la lista, en formato CSV válido."
                messages.append({"role": "user", "content": newQuery})
                newResponse = self.analyzeText(messages, chosenModel)
                messages.append({"role": "assistant", "content": newResponse})
                tryCounter += 1
            # Si el campo que debe ser año incluye texto no numera, generar una nueva query y analizarla
            elif not newResponseSplit[0].isdigit():
                newQuery = "Me diste una respuesta incorrecta, pues me contestaste con algo que no te pedí. Responde de nuevo pero SOLAMENTE con el año_del_documento y el tipo_de_documento proveniente de la lista, en formato CSV válido."
                messages.append({"role": "user", "content": newQuery})
                newResponse = self.analyzeText(messages, chosenModel)
                messages.append({"role": "assistant", "content": newResponse})
                tryCounter += 1
            else:
                queryExitoso = True
    
        return newResponse
    
    def analyzeText(self, messages, chosenModel):
        response = openai.ChatCompletion.create(
            model=chosenModel,
            temperature=0.5,
            messages=messages
        )
        return response["choices"][0]["message"]["content"]

    def cleanText(self, text):
        
        return text
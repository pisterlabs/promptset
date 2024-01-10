import json

import openai
import secretKeys
from episodi.wildBytes_1_7.ex7_1_CommonFunction import CommonFunctions
from episodi.wildBytes_1_7.functionManager import OpenAIApiFunctionManager


class ChatGptAPIWrapper:
    promptTokens = 0  # tokens utilizzati per il prompt 0.0015 dollari
    completionTokens = 0  # tokens utilizzati per la completion 0.002 dollari
    totalTokens = 0
    budget = 0.05  # 5 centesimi di dollaro
    chatHistory = []
    functionManager: OpenAIApiFunctionManager
    models = {"gpt-4", "gpt-3.5-turbo", "babbage-002", "davinci-002",
              "text-davinci-003", "text-davinci-002", "davinci", "curie", "babbage", "ada"}
    _model = "gpt-3.5-turbo"
    isWannaExit = False
    functions = []
    _temperature = 0
    _max_tokens = 50

    def __init__(self, parent):
        openai.api_key = secretKeys.openAi
        self.commonFunctions = CommonFunctions()
        self.initVariables()
        self.initHistory()
        self.initFunctions()
        self.parent = parent

    #                   INIT FUNCTIONS

    def initVariables(self):
        # inizializza le variabili che contengono i tokens
        data = self.open("data.json")
        self.deserializeData(data)

    def initHistory(self):
        # inizializza la chat history
        data = self.open("history.json")
        self.deserializeHistory(data)

    def initFunctions(self):
        # inizializza le funzioni
        self.functionManager = OpenAIApiFunctionManager()
        content_param = self.functionManager.createContentParameter("string", "Informazioni da memorizzare")
        parameters = self.functionManager.createParameter(content_param, ["content"])
        self.functionManager.addFunction("addToMemory", "Aggiunge informazioni alla memoria", parameters)

    #                   TOKENS FUNCTIONS

    def remainingTokens(self):
        """
        Calcola i tokens rimanenti partendo dal costo per 1000 tokens.
        :return:
        """
        # Costo per 1000 tokens
        costPer_1000_tokens = 0.002
        costPerToken = costPer_1000_tokens / 1000
        self.budget = self.budget - (costPerToken * self.totalTokens)
        remaining_tokens = float(self.budget / costPerToken)
        self.serialize()
        return f"Ti rimangono ${self.budget:.4f}. Puoi ancora utilizzare {int(remaining_tokens)} tokens."

    def addTokensFromResponse(self, response):
        self.promptTokens += int(response['usage']['prompt_tokens'])
        self.completionTokens += int(response['usage']['completion_tokens'])
        self.totalTokens += int(response['usage']['total_tokens'])
        print("Tokens aggiunti")
        print(
            f"Prompt tokens: {self.promptTokens} Completion tokens: {self.completionTokens} Total tokens: {self.totalTokens}")

    def askToAddMoreBudget(self):
        """
        Chiede all'utente se vuole aumentare il budget.
        :return:
        """
        answer = input("Vuoi aumentare il budget? [Y/n]")
        if answer.lower() != "y":
            exit(0)
        self.coinUp()
        return True

    def coinUp(self):
        """
        Aggiunge 5 centesimi di dollaro al budget.
        """
        newBudget = input("Inserisci il nuovo budget: ")
        self.budget += float(newBudget)
        self.promptTokens = 0
        self.completionTokens = 0
        self.totalTokens = 0
        self.serialize()

    #                   MODEL GET SET FUNCTIONS

    def getModel(self):
        return self._model

    def setModel(self, value):
        self._model = value

    def getTemperature(self):
        return self._temperature

    def setTemperature(self, value):
        self._temperature = value

    def getMaxTokens(self):
        return self._max_tokens

    def setMaxTokens(self, value):
        self._max_tokens = value

    def setMemory(self, content):
        """Aggiunge informazioni alla memoria"""
        self.parent.addMemoriesFromGpt(content)
        # E aggiungi la risposta al tuo chatHistory
        self.chatHistory.append({"role": "assistant", "content": f"Added to memory: {content}"})
        return content

    #                   CHECK FUNCTIONS

    def checkQuestion(self, question):
        if type(question) is str:
            messages = [
                {"role": "user", "content": f"{question}"},
            ]
        elif type(question) is list:
            messages = question
        else:
            return "Errore: la domanda deve essere una stringa o una lista di messaggi."
        self.chatHistory.append({"role": "user", "content": f"{question}"})
        return messages

    def checkFunctionCall(self, response):
        try:
            # Estrai il primo choice e il suo contenuto
            first_choice = response.get('choices', [{}])[0]
            message = first_choice.get('message', {})
            function_call = message.get('function_call', {})

            # Estrai nome funzione e argomenti
            function_name = function_call.get('name')
            arguments = json.loads(function_call.get('arguments', '{}'))

            # Controllo per addToMemory
            if function_name == 'addToMemory':
                content_to_add = arguments.get('content')
                answer = self.setMemory(content_to_add)
                return f"Added to memory: {answer}"
            else:
                self.chatHistory.append(
                    {"role": "assistant", "content": "Function not recognized"}
                )
                return f"Errore: la funzione {function_name} non è riconosciuta."
        except Exception as e:
            print(response)
            return f"Errore: {e}"

    #                   CHAT GPT FUNCTIONS

    def getResponse(self, messages):
        return openai.ChatCompletion.create(
            model=self.getModel(),
            messages=messages,
            max_tokens=self.getMaxTokens(),  # limita la lunghezza della risposta
            temperature=self.getTemperature(),  # 0 = risposta più probabile, 1 = risposta più creativa
            functions=self.functionManager.functions,
            function_call="auto",
        )

    def getAnswer(self, question):
        """
        Ottiene una risposta da openAI. Ogni risposta costa 0.002 dollari.
        Usa la funzione checkQuestion per controllare se la domanda è una stringa o una lista di messaggi.
        Quindi usa la funzione getResponse per ottenere la risposta da openAI.
        :param question: la domanda da porre al chatbot.
        :return:
        """
        messages = self.checkQuestion(question)
        if "Errore" in messages:
            return messages
        manager = OpenAIApiFunctionManager()
        content_param = manager.createContentParameter("string", "Informazioni da memorizzare")
        parameters = manager.createParameter(content_param, ["content"])
        manager.addFunction("addToMemory", "Aggiunge informazioni alla memoria", parameters)
        response = self.getResponse(messages)
        if isinstance(response, str):
            return f"Errore: {response}"
        answer = response['choices'][0]['message']['content']
        self.addTokensFromResponse(response)
        if 'function_call' in response['choices'][0]['message']:
            self.checkFunctionCall(response)
        else:
            # Altrimenti, semplicemente aggiungi la risposta normale al tuo chatHistory
            self.chatHistory.append({"role": "assistant", "content": f"{answer}"})

        return f"Risposta: {answer}\n{self.remainingTokens()}"

    #                   FILE FUNCTIONS

    def open(self, fileName):
        """
        Apre un file e ne ritorna il contenuto.
        :param fileName:
        :return:
        """
        try:
            return self.commonFunctions.read(fileName)
        except FileNotFoundError:
            print(f"I dati del file: {fileName} Non sono stati trovati")
            answer = input("Vuoi che creo un nuovo file? [Y/n]")
            if answer.lower() == "y":
                self.serialize()
                return self.open(fileName)
            else:
                print("Exiting...")
                exit(1)

    def serialize(self):
        """
       Serializza i dati in un file json.
       :return: i dati in formato json
       """
        data = {
            "promptTokens": self.promptTokens,
            "completionTokens": self.completionTokens,
            "totalTokens": self.totalTokens,
            "budget": self.budget,
        }
        jsonData = json.dumps(data, indent=4)
        # salva i valori dei tokens
        self.commonFunctions.write("data.json", str(jsonData))
        # salva la chat history
        jSonHistory = json.dumps(self.chatHistory, indent=4)
        self.commonFunctions.write("history.json", str(jSonHistory))
        return data

    def deserializeData(self, data=None):
        """
       Deserializza i dati da un file json.
       :return: i dati in formato json
       """
        jsonData = json.loads(data)
        self.promptTokens = jsonData["promptTokens"]
        self.completionTokens = jsonData["completionTokens"]
        self.totalTokens = jsonData["totalTokens"]
        self.budget = jsonData["budget"]
        return jsonData

    def deserializeHistory(self, data):
        """
        Deserializza la chat history da un file json.
        :param data:
        :return:
        """
        jsonData = json.loads(data)
        self.chatHistory = jsonData
        return jsonData


if __name__ == '__main__':
    ai = ChatGptAPIWrapper()

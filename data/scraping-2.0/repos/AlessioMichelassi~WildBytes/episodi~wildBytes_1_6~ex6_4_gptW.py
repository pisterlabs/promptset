import json

import openai
import secretKeys


class ChatGptAPIWrapper:
    promptTokens = 0  # tokens utilizzati per il prompt 0.0015 dollari
    completionTokens = 0  # tokens utilizzati per la completion 0.002 dollari
    totalTokens = 0
    budget = 0.05  # 5 centesimi di dollaro
    chatHistory = []
    models = {"gpt-4", "gpt-3.5-turbo", "babbage-002", "davinci-002",
              "text-davinci-003", "text-davinci-002", "davinci", "curie", "babbage", "ada"}
    model = "gpt-3.5-turbo"
    isWannaExit = False

    def __init__(self, parent):
        openai.api_key = secretKeys.openAi
        self.initVariables()
        self.initHistory()
        self.parent = parent

    def initVariables(self):
        # inizializza le variabili che contengono i tokens
        data = self.open("data.json")
        self.deserializeData(data)

    def initHistory(self):
        # inizializza la chat history
        data = self.open("history.json")
        self.deserializeHistory(data)

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

    def addToMemory(self, content):
        """Aggiunge informazioni alla memoria"""
        self.parent.addMemoriesFromGpt(content)
        print(f"sto aggiungendo alla memoria: {content}")
        return content

    def getAnswer(self, question):
        """
        Ottiene una risposta da openAI. Ogni risposta costa 0.002 dollari.
        :param question:
        :return:
        """
        if type(question) is str:
            messages = [
                {"role": "user", "content": f"{question}"},
            ]
        elif type(question) is list:
            messages = question
        else:
            return "Errore: la domanda deve essere una stringa o una lista di messaggi."
        self.chatHistory.append({"role": "user", "content": f"{question}"})
        functions = [
            {
                "name": "addToMemory",
                "description": "Aggiunge informazioni alla memoria",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Informazioni da memorizzare",
                        }
                    },
                    "required": ["content"],
                },
            }
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=50,  # limita la lunghezza della risposta
            temperature=0,  # 0 = risposta più probabile, 1 = risposta più creativa
            functions=functions,
            function_call="auto",
        )
        if isinstance(response, str):
            return f"Errore: {response}"
        answer = response['choices'][0]['message']['content']
        self.promptTokens += int(response['usage']['prompt_tokens'])
        self.completionTokens += int(response['usage']['completion_tokens'])
        self.totalTokens += int(response['usage']['total_tokens'])

        if 'function_call' in response['choices'][0]['message'] and response['choices'][0]['message']['function_call'][
            'name'] == 'addToMemory':
            try:
                arguments = json.loads(response['choices'][0]['message']['function_call']['arguments'])
                content_to_add = arguments['content']

                self.addToMemory(content_to_add)
                # E aggiungi la risposta al tuo chatHistory
                self.chatHistory.append({"role": "assistant", "content": f"Added to memory: {content_to_add}"})
            except Exception as e:
                print(response)
                return f"Errore: {e}"
        else:
            # Altrimenti, semplicemente aggiungi la risposta normale al tuo chatHistory
            self.chatHistory.append({"role": "assistant", "content": f"{answer}"})

        return f"Risposta: {answer}\n{self.remainingTokens()}"

    def getSecondAnswer(self, answer, messages):
        return f"{answer}\n{messages}"

    def open(self, fileName):
        """
        Apre un file e ne ritorna il contenuto.
        :param fileName:
        :return:
        """
        try:
            with open(fileName, 'r', encoding="utf-8") as file:
                data = file.read()
            return data
        except FileNotFoundError:
            print(f"I dati del file: {fileName} Non sono stati trovati")
            answer = input("Vuoi che creo un nuovo file? [Y/n]")
            if answer.lower() == "y":
                self.serialize()
                return self.open(fileName)
            else:
                print("Exiting...")
                exit(1)

    def save(self, fileName, data):
        """
        Salva un file.
        :param fileName: il nome del file
        :param data: il contenuto
        """
        with open(fileName, 'w', encoding="utf-8") as file:
            file.write(data)

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
        self.save("data.json", str(jsonData))
        # salva la chat history
        jSonHistory = json.dumps(self.chatHistory, indent=4)
        self.save("history.json", str(jSonHistory))
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



if __name__ == '__main__':
    ai = ChatGptAPIWrapper()

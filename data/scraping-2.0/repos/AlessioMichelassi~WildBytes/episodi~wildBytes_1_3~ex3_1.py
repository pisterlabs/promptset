import json

import openai
import secretKeys


class ChatGptAPIWrapper:
    promptTokens = 0  # tokens utilizzati per il prompt 0.0015 dollari
    completionTokens = 0  # tokens utilizzati per la completion 0.002 dollari
    totalTokens = 0
    budget = 0.05  # 5 centesimi di dollaro
    chatHistory = []
    memory = "MEMORIES: "
    models = {"gpt-4", "gpt-3.5-turbo", "babbage-002", "davinci-002",
              "text-davinci-003", "text-davinci-002", "davinci", "curie", "babbage", "ada"}
    model = "gpt-3.5-turbo"
    isWannaExit = False

    def __init__(self):
        openai.api_key = secretKeys.openAi
        self.initVariables()
        self.initHistory()
        self.startChat()

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

    def getAnswer(self, question):
        """
        Ottiene una risposta da openAI. Ogni risposta costa 0.002 dollari.
        :param question:
        :return:
        """
        messages = [
            {"role": "user", "content": f"{self.memory}, Question: {question}"},
        ]
        self.chatHistory.append({"role": "user", "content": f"{question}"})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=50,  # limita la lunghezza della risposta
            temperature=0,  # 0 = risposta più probabile, 1 = risposta più creativa
        )
        answer = response['choices'][0]['message']['content']
        self.promptTokens += int(response['usage']['prompt_tokens'])
        self.completionTokens += int(response['usage']['completion_tokens'])
        self.totalTokens += int(response['usage']['total_tokens'])
        self.chatHistory.append({"role": "assistant", "content": f"{answer}"})
        return f"Risposta: {answer}\n{self.remainingTokens()}"

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

    def startChat(self):
        while self.isWannaExit is False:
            question = input("Domanda: ").strip()  # Utilizza strip() per rimuovere spazi bianchi all'inizio e alla fine
            if not question:  # Controlla se la domanda è vuota
                print("Per favore, inserisci una domanda valida.")
                continue
            if question.lower() == "#exit":
                self.isWannaExit = True
                break
            elif question.lower() == "#memorize":
                self.addMemories(input("Cosa vuoi ricordare? "))
                continue
            if self.budget <= 0.002:
                print("Non hai abbastanza soldi per continuare a chattare.")
                self.isWannaExit = self.askToAddMoreBudget()
            print(self.getAnswer(question))

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

    def addMemories(self, value):
        """
        Aggiunge un ricordo alla memoria.
        :param value:
        :return:
        """
        self.memory += f"{value}, "


if __name__ == '__main__':
    ai = ChatGptAPIWrapper()

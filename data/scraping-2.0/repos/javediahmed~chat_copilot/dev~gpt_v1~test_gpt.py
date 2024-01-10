import openai
import os
import json
from datetime import datetime

API_KEY = os.getenv("OPENAI_API_KEY")

MODELS = {
  "GPT-1": "text-gpt-1-en-12b",
  "GPT-2": "text-gpt-2-en-117b",
  "GPT-3": "text-davinci-002",
  "GPT-3.5": "text-davinci-003",
  "GPT-4": "text-davinci-004",
  "Jurassic-1 Jumbo": "text-jurassic-1-jumbo-en-175b",
  "Megatron-Turing NLG": "text-megatron-turing-nlg-345m-355b",
  "WuDao 2.0": "text-wudao-2-0-en-1.76T"
}

class ChatGPT:
    def __init__(self):
        self.api_key = API_KEY
        openai.api_key = self.api_key
        self.model = MODELS["GPT-3"]
        self.messages = [{"role": "system", "content": "You are talking to GPT-3"}]
        self.history = []

    def get_user_input(self, prompt):
        user_input = input(prompt).strip()
        if user_input.lower() in ["exit", "x"]:
            print("Exiting script.")
            exit(0)
        return user_input

    def process_query_input(self, source='manual'):
        if source == 'manual':
            query = self.get_user_input("Enter your query: ")
            self.history.append({"role": "user", "content": query})
        else:
            filename = self.get_user_input("Enter the file path: ")
            with open(filename, 'r') as file:
                query_dict = json.load(file)
                query = query_dict.get('content', '')
                self.history.append(query_dict)

        return query

    def process_settings_input(self):
        print("Current Model: ", [key for key, value in MODELS.items() if value == self.model][0])
        new_model = self.get_user_input("Enter a new model name or hit enter to keep the current model: ")
        if new_model != '' and new_model in MODELS:
            self.model = MODELS[new_model]
        print("Model updated to: ", [key for key, value in MODELS.items() if value == self.model][0])

    def save_history(self):
        filename = self.get_user_input("Enter the file name to save history: ")
        with open(filename, 'w') as file:
            json.dump(self.history, file)
        print(f"History saved in {filename}")

    def chat(self, query):
        self.messages.append({"role": "user", "content": query})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            max_tokens=100
        )
        return response['choices'][0]['message']['content']

if __name__ == "__main__":
    print("Test GPT interface")
    print("Today's date: ", datetime.today().strftime('%Y-%m-%d'))
    chatGPT = ChatGPT()

    while True:
        print("\nMain Menu:")
        print("1. Enter a query")
        print("2. Load query from a file")
        print("3. Settings")
        print("4. Save history")
        print("5. Exit")
        user_option = chatGPT.get_user_input("Enter your choice (1 to 5): ")

        if user_option == "1":
            query = chatGPT.process_query_input(source='manual')
            if query:
                response = chatGPT.chat(query)
                print("\nGPT-3 Response:")
                print(response)
        elif user_option == "2":
            query = chatGPT.process_query_input(source='file')
            if query:
                response = chatGPT.chat(query)
                print("\nGPT-3 Response:")
                print(response)
        elif user_option == "3":
            chatGPT.process_settings_input()
        elif user_option == "4":
            chatGPT.save_history()
        elif user_option == "5":
            print("Exiting script.")
            exit(0)
        else:
            print("Invalid choice. Please select again.")

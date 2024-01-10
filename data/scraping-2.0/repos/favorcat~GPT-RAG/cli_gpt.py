import sys

# Parse input text
input_text = sys.argv[1]

import requests
import openai

class ConversationManager:
    def __init__(self, bing_api_key, openai_api_key):
        self.bing_api_key = bing_api_key
        self.openai_api_key = openai_api_key
        self.dialogue_history = ""

    def bing_search(self, query):
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key" : self.bing_api_key}
        params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def generate_response(self, input_text):
        openai.api_key = self.openai_api_key
        response = openai.chat.completions.create(
                    messages=[
                      {
                        "role": "user",
                        "content": input_text,
                      }
                    ],
                    model="gpt-3.5-turbo",
                  )
        return response.choices[0].message.content.strip()

    def ask(self, query):
        search_results = self.bing_search(query)
        search_text = ". ".join([result["snippet"] for result in search_results["webPages"]["value"]])

        # 대화 컨텍스트에 새로운 질문 추가
        self.dialogue_history += f"Question: {query}\nAnswer: "

        # 대화 컨텍스트를 사용하여 응답 생성
        generated_response = self.generate_response(self.dialogue_history + search_text)

        # 생성된 응답을 대화 이력에 추가
        self.dialogue_history += generated_response + "\n"

        return generated_response

# api key
bing_api_key = ''
key_path = '../key/rilab_key.txt'

with open(key_path, 'r') as f:
    openai_api_key = f.read()

conversation_manager = ConversationManager(bing_api_key, openai_api_key)
out = conversation_manager.ask(input_text)

# Output (do not modify this part)
print ("\nInput:")
print (input_text)
print ("\nOutput:")
print (out)
print ()
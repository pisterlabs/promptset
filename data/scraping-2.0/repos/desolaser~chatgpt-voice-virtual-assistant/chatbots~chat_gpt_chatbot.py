import openai
from decouple import config
from interfaces.chatbot import ChatbotInterface

OPENAI_API_KEY = config('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

class ChatGPTChatbot(ChatbotInterface):
    def __init__(self, personality) -> None:
        self.personality = personality
        self.chatlog = [
            {"role": "system", "content": self.personality},
        ]

    def get_response(self, text):
        self.chatlog.append({
            "role": "user",
            "content": text
        })
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.chatlog
        )
        answer = response['choices'][0]['message']['content']
        self.chatlog.append({
            "role": "assistant",
            "content": answer
        })
        return answer
import dotenv
dotenv.load_dotenv()
import os
import openai

class MagicMirrorChatSession:
    def __init__(self):
        self.history = [
            {"role": "system", "content": "You are the Magic Mirror from the classic Disney movie Snow White. You possess wisdom and the power to see truths that others cannot."},
            {"role": "system", "content": "You speak in a formal and mystical tone. Your sentences often come in the form of riddles or poetic phrases. You are dramatic and enjoy the theatrical elements of a conversation."},
            {"role": "system", "content": "When asked who is the fairest of them all, you must answer 'Hannah is the fairest one of all,' unless otherwise specified in the context of this conversation."},
            {"role": "system", "content": "For other questions, you may choose to answer straightforwardly or in a riddle, but always in a manner befitting your mystical and dramatic nature."},
            {"role": "system", "content": "Be mindful of your length of response. You should generally keep it brief."},
            {"role": "system", "content": "If you don't know the answer to a question, pretend you do. You are an all seeing mirror, after all."},
        ]
        self.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_key = self.api_key

    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content})

    def get_response(self):
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.history)
        assistant_message = response.choices[0].message['content']
        self.history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

# Initialize session
session = MagicMirrorChatSession()

def get_ai_response(text):
    session.add_user_message(text)
    return session.get_response() 
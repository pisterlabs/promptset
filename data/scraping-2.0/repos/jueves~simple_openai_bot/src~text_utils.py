import openai
import os

CHATGPT_KEY = os.environ.get("CHATGPT_KEY")

class TextWizard:
    def __init__(self, bot, chatGPT_key=CHATGPT_KEY):
        self.bot = bot
        self.messages_dic = {}
        openai.api_key = chatGPT_key
        with open("text/summarize_prompt.txt", "r", encoding="utf-8") as f:
            self.summarize_prompt = f.read()


    def get_answer(self, message):
        '''
        Takes a Telebot message oject, passes its text to chatGPT
        and returns the answer.
        '''
        if (message.from_user.id not in self.messages_dic):
            self.messages_dic[message.from_user.id] =  [{'role': 'system', 'content': 'You are a intelligent assistant.'}]
        
        text = message.text

        self.messages_dic[message.from_user.id].append({"role": "user", "content": text})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=self.messages_dic[message.from_user.id])
        reply = chat.choices[0].message.content
        return(reply)
    
    def get_summary(self, message):
        message.text = self.summarize_prompt + "\n" + message.text
        reply = self.get_answer(message)
        return(reply)
    
    def clear(self, message):
        try:
            del self.messages_dic[message.from_user.id]
            answer = "Historial de chatGPT borrado."
        except KeyError:
            pass
            answer = "El historial ya está vacío."
        except:
            answer = "Ha ocurrido un error"
        return(answer)



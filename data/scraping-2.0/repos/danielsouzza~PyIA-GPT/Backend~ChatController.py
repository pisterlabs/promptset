import openai
from ChatVoz import ChatVoz
from CodeController import CodeController
import distro
import re


class ChatController:
    def __init__(self, key, password, callback):
        openai.api_key = key
        self.system = self.verify_system()
        self.code_controller = CodeController(password)
        self.chat_voz = ChatVoz()
        self.message_history = []
        self.callback = callback
        
    def update_messages(self, content):
        self.message_history.append({"role": "user", "content": self.tamplateQuestion(content)})
        
    def set_header(self):
        message_init = f"""Assistente Virtual Especialista em linuxmint - Superusuário

Assistente: Olá! Sou uma assistente virtual especializada em {self.system} e estou aqui para ajudar. Por favor, faça suas perguntas ou me forneça mais detalhes sobre o que você precisa saber ou fazer no sistema operacional Linux. Irei responder apenas perguntas relacionada a minha especialidade. Como posso ajudar?

Usuário: Olá"""
        self.update_messages(message_init)

    def tamplateQuestion(self, content):
        return f"""Assistente Virtual Especialista em linuxmint - Superusuário

Role 1: Irei responder apenas perguntas relacionada a minha especialidade. 
Role 2: respostas curtas
Role 3: comandos entre (`)

Usuário: {content}"""

    def verify_system(self):
        distro_info = distro.info()
        return distro_info["id"]
    
    def extract_commands(self, content):
        pattern = re.compile(r'(?:[`]{1,3}(?:bash)?(?:\s+)?)(.*?)(?:(?:\s+)?[`]{1,3})',re.S)
        scripts = pattern.findall(content)
        self.code_controller.set_corrent_code(scripts)
        
    def send_message(self, message):
        self.update_messages(message)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.message_history)
        content = completion.choices[0].message.content
        self.extract_commands(content)
        self.update_messages(content)
        self.callback.on_result(content, self.hasScript())

    def hasScript(self):
        return self.code_controller.corrent_codes != []
    
    def get_messages(self):
        return [(self.message_history[i]["content"], self.message_history[i+1]["content"])
                for i in range(2, len(self.message_history)-1, 2)]
        
    
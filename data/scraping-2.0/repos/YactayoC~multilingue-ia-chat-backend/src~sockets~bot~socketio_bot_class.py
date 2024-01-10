from cryptography.fernet import Fernet
from flask_socketio import emit
import unicodedata
import openai
import os


from src.services.common.message_service import MessageService
from src.services.common.aws_translate import translate_text


class MessageBotHandler:
    api_key = os.getenv("API_KEY")
    context = ""
    context_initialized = False

    def __init__(self, socketio_instance):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
        self.private_rooms = {}
        self.socketio = socketio_instance
        self.message_service = MessageService()
        # self.context = self.read_context_from_file(self.context_file, "Precio")

    def encrypt(self, message):
        encrypted_message = self.cipher_suite.encrypt(message.encode())
        return encrypted_message.decode("ascii")

    def decrypt(self, encrypted_message):
        decrypted_message = self.cipher_suite.decrypt(encrypted_message.encode("ascii"))
        decrypted_message = decrypted_message.decode("utf-8")
        decrypted_message = unicodedata.normalize("NFKD", decrypted_message)
        return decrypted_message

    def read_context_from_file(self, areaProcess=""):
        if areaProcess == "analitica_web":
            self.context_file = "analitica_web.txt"

        if areaProcess == "email_marketing":
            self.context_file = "email_marketing.txt"

        if areaProcess == "marketing_contenidos":
            self.context_file = "marketing_contenidos.txt"

        if areaProcess == "publicidad_linea":
            self.context_file = "publicidad_linea.txt"

        if areaProcess == "redes_sociales":
            self.context_file = "redes_sociales.txt"

        if areaProcess == "seo":
            self.context_file = "seo.txt"

        with open(self.context_file, "r") as file:
            return file.read()

    def interact_with_chatgpt(self, data):
        message = data["message"]
        room_name = data["room_name"]
        areaProcess = data["areaProcess"]
        code_language = data["code_language"]
        message = message.lower()

        self.context = self.read_context_from_file(areaProcess)

        prompt = self.context + "\n" + message
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            api_key=self.api_key,
        )

        # context = response.choices[0].text.encode("cp1252").decode()
        context = response.choices[0].text.strip()
        context_translated = translate_text(context, target=code_language)
        print(context)
        self.socketio.emit(
            "get_messages_gpt",
            {"room_name": room_name, "message": context_translated, "user": "Bot"},
            room=room_name,
        )

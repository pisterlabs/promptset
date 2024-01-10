from src.va.mongo.connection_factory import ConnectionFactory
from src.va.flaskr import app
from flask import current_app
from src.va.openai_tools.ai_chat import OpenAIChat
from src.va.services.error import InvalidKeyError

class Service:
    def __init__(self):
        with app.app_context():
            self.mongo_config = current_app.config["mongo"]
            self.system_config = current_app.config["system"]
            self.filepath = current_app.config["UPLOAD_FOLDER"]
            self.factory = ConnectionFactory(self.mongo_config["host"],
                                        int(self.mongo_config["port"]),
                                        self.mongo_config["username"],
                                        self.mongo_config["password"])

    def validate_openai_message_keys(self, messages: list[dict]):
        valid_keys = [OpenAIChat.ROLE, OpenAIChat.CONTENT, OpenAIChat.SYSTEM, OpenAIChat.USER, OpenAIChat.ASSISTANT]
        for message in messages:
            check = all(item in valid_keys for item in list(message.keys()))
            if not check:
                raise InvalidKeyError("Some keys are not supported by openai APIs")

    def validate_context_fields(self, key_list: list):
        context_fields = ["config", "chat_model", "stt_model", "token_limit", "messages", "default"]
        check = all(item in context_fields for item in key_list)
        if not check:
            raise InvalidKeyError("Some keys are not supported by context object")

    def str2bool(self, string: str) -> bool:
        if string is None:
            return False
        return string.lower() in ("true", "1", "yes")
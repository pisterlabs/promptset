from chat.chat_history import ChatHistory, MessageType
from chat.openAI_client import OpenAIClient


class ArtificialFriend:
    def __init__(self, identity_message, model_name='gpt-4-1106-preview',
                 agent_name="assistant", chat_mode="solo"):
        self.agent_name = agent_name
        self.chat_mode = chat_mode
        self.openai_client = OpenAIClient.getInstance()
        self.model_name = model_name
        self.messageHistory = ChatHistory()
        self.identity_message = identity_message
        self.set_identity(identity_message)

    def __add_message(self, message, role="assistant", message_type=MessageType.Correspondence):
        self.messageHistory.append_message(role, self.agent_name, message, message_type=message_type)

    def set_identity(self, identity_message_text):
        self.identity_message = identity_message_text
        history = self.messageHistory.get_raw_message_list()
        identity_message = None
        for message in history:
            if message.message_type == MessageType.Identity and message.username == self.agent_name:
                identity_message = message
                break
        if identity_message is not None:
            identity_message.set_message_text(identity_message_text)
        else:
            self.__add_message(identity_message_text, "system", message_type=MessageType.Identity)

    def clear_messages(self):
        self.messageHistory.clear_messages()
        self.set_identity(self.identity_message)

    def send_message(self, message):
        return self.send_message_solo(message)

    def send_message_solo(self, message):
        self.__add_message(message, "user")
        message_list = self.messageHistory.get_openai_message_list()
        response = self.openai_client.call_chat_completion(self.model_name, message_list)
        self.__add_message(response)
        return response

    def insert_fake_agent_message(self, message):
        self.__add_message(message)

    def insert_fake_user_message(self, message):
        self.__add_message(message, "user")

    def insert_fake_system_message(self, message):
        self.__add_message(message, "system")

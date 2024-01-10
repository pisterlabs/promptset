from api.open_ai_llm import OpenAiLlmApi, OpenAiLlmOptions, OpenAiChatMessages
from entities.conversation import Conversation, UserChatMessage
from config.main import openai_default_model, openai_default_temperature, openai_max_message_history_length
from entities.command_line_app import CommandLineApp


class CommandLineEventLoop:
    def __init__(self, command_line_app):
        self.command_line_app = command_line_app

    def start(self, tick):
        while True:
            tick()


class GptChatCommandLineApp(CommandLineApp):
    def __init__(self, few_shot_messages=[]):
        self.openai_llm_api = OpenAiLlmApi(OpenAiLlmOptions(openai_default_model, float(openai_default_temperature)))
        self.conversation = Conversation(few_shot_messages)

    def start(self):
        event_loop = CommandLineEventLoop(self)
        event_loop.start(self.chat)

    def chat(self):
        user_input = input("You: ")
        chat_response_text = self._get_chat_response_text(user_input)
        self._print_chat_response(chat_response_text)
        self.conversation.add_message(chat_response_text)
        self._truncate_message_history()

    def _print_chat_response(self, chat_response_text):
        print("AI: " + chat_response_text)

    def _truncate_message_history(self):
        if (len(self.conversation.state.messages) > int(openai_max_message_history_length)):
            self.conversation.pop_message()

    def _get_chat_response_text(self, user_input):
        self.conversation.add_message(UserChatMessage(user_input))
        messages = self.conversation.state.messages
        chat_response = self.openai_llm_api.chat_completion(
            chat_messages=OpenAiChatMessages(messages),
        )
        return chat_response.choices[0].message.content

from app.gpt_intention_detection import GptIntentionDetection
from api.open_ai_llm import OpenAiLlmApi, OpenAiLlmOptions, OpenAiChatMessages
from config.main import (
    openai_default_model,
    openai_default_temperature,
    openai_max_message_history_length,
)
from entities.conversation import Conversation, UserChatMessage
from entities.intention import Intention
from entities.command_line_app import CommandLineApp
from entities.command_line_event_loop import CommandLineEventLoop


intention_descriptions = {
    "accept": "accepts, confirms, or agrees with the previous message",
    "reject": "rejects, declines, or disagrees with the previous message",
    "exit": "exits the conversation",
    "answer": "answers a question",
    "none": "none of the other intentions apply",
}


class GptBuddyCommandLineApp(CommandLineApp):
    def __init__(self, few_shot_messages=[]):
        self.openai_llm_api = OpenAiLlmApi(
            OpenAiLlmOptions(openai_default_model, float(openai_default_temperature))
        )
        self.conversation = Conversation(few_shot_messages)
        self.intentions = [
            Intention(intention_name, intention_description)
            for intention_name, intention_description in intention_descriptions.items()
        ]

    def start(self):
        event_loop = CommandLineEventLoop(self)
        event_loop.start(self.handle_input)

    def handle_input(self):
        user_input = input("You: ")
        self.conversation.add_message(UserChatMessage(user_input))
        user_intention = self._get_user_intention()
        self._process_user_intention(user_intention)

    def _process_user_intention(self, user_intention):
        response_text = "hmm... I don't understand what you mean. Please try again."
        if user_intention == "answer":
            response_text = self._get_chat_response_text()
        elif user_intention == "none":
            response_text = self._get_chat_response_text()
        elif user_intention == "exit":
            exit()

        self._print_chat_response(response_text)

    def _get_chat_response_text(self):
        messages = self.conversation.state.messages
        chat_response = self.openai_llm_api.chat_completion(
            chat_messages=OpenAiChatMessages(messages),
        )
        return chat_response.choices[0].message.content

    def _print_chat_response(self, chat_response_text):
        print("AI: " + chat_response_text)

    def _truncate_message_history(self):
        if len(self.conversation.state.messages) > int(
            openai_max_message_history_length
        ):
            self.conversation.pop_message()

    def _get_user_intention(self):
        intention_detection = GptIntentionDetection(
            self.conversation.state.messages, self.intentions
        )
        return intention_detection.get_intention_of_last_message()

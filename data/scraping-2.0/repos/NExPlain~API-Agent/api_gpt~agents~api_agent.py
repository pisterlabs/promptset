from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_to_dict


class ApiAgent:
    def __init__(
        self,
        human_prefix="User",
        ai_prefix="APIBot",
    ):
        # Initialize an instance of the ApiAgent class with default or user-defined human and AI prefixes.
        self.memory = ConversationBufferMemory(
            human_prefix=human_prefix, ai_prefix=ai_prefix
        )

    def add_user_message(self, message: str):
        # Add a user message to the conversation memory.
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        # Add an AI message to the conversation memory.
        self.memory.chat_memory.add_ai_message(message)

    def format_ai_message_for_display(self, message: str):
        # Remove the "NEED_MORE_INFORMATION: " prefix from the AI message for improved readability.
        return message.replace("NEED_MORE_INFORMATION: ", "")

    def clear_memory(self):
        # Clear the conversation memory, allowing for a fresh conversation.
        self.memory.clear()

    def get_conversational_history(self) -> str:
        """Example:

        [{'type': 'human', 'data': {'content': 'hi!', 'additional_kwargs': {}}},
         {'type': 'ai', 'data': {'content': 'whats up?', 'additional_kwargs': {}}}]

          Returns:
              str: A formatted string representing the conversational history.
        """
        dicts = messages_to_dict(self.memory.chat_memory.messages)
        ret = ""
        for dict in dicts:
            ret += f"{dict['type']}: {dict['data']['content']}\n"
        return ret

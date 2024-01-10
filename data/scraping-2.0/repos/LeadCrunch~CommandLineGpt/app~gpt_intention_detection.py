# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from entities.conversation import SystemChatMessage, UserChatMessage
from entities.intention import Intention
from api.open_ai_llm import OpenAiChatMessages, OpenAiLlmApi, OpenAiLlmOptions

system_prompt_template = """
consider the conversation below, which is delimeted by three backticks:
```
{messages}
```
given the conversation above, whenever the user says anything, find the intention that best matches the user's message.
please use the following intentions, which are delimeted by three backticks:
```
{intentions}
```
"""


default_chat_intentions = [
    Intention("accept", "confirms, accepts, or agrees with the previous message"),
    Intention("modify_plan", "user wants to change something about the plan"),
]


def build_user_intention_function(intentions):
    return {
        "name": "get_user_intention",
        "description": "Determine the users intention based on the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "intention": {"type": "string", "enum": [intention.name for intention in intentions]},
            },
            "required": ["intention"],
        },
    }


class GptIntentionDetection:
    def __init__(self, messages, intentions=default_chat_intentions):
        self.messages = messages
        self.intentions = intentions
        self.llm_api = OpenAiLlmApi(OpenAiLlmOptions("gpt-3.5-turbo-0613", 0))

    def get_intention_of_last_message(self):
        chat_completion_prompt_messages = self._chat_completion_prompt_messages()

        function_call_arguments = self.llm_api.function_call(
            chat_messages=OpenAiChatMessages(chat_completion_prompt_messages),
            functions=[
                build_user_intention_function(self.intentions),
            ],
            function_call={"name": "get_user_intention"},
        )

        return function_call_arguments["intention"]

    def _chat_completion_prompt_messages(self):
        system_message = SystemChatMessage(self._system_message_text())
        user_message = UserChatMessage(self.messages[-1].text)
        return [system_message, user_message]

    def _system_message_text(self):
        formatted_messages = ["{}: {}, \n".format(message.role, message.text) for message in self.messages]
        formatted_intentions = [
            "name: {}, description: {} \n".format(intention.name, intention.description)
            for intention in self.intentions
        ]
        system_message_text = system_prompt_template.format(
            messages="\n".join(formatted_messages),
            intentions="\n".join(formatted_intentions),
            example_intention_name=self.intentions[0].name,
        )

        return system_message_text

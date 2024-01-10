import openai

from .taifun import Taifun

from rich.console import Console

from taifun.openai_api import FunctionCall, Message, Role

import json


class TaifunConversationRunnerException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class TaifunConversationRunner:
    def __init__(self, taifun: Taifun, openai_model: str = "gpt-3.5-turbo-0613"):
        self.taifun = taifun
        self.openai_model = openai_model

        self.__end_of_conversation = False
        self.__end_conversation_reason: str | None = None

        taifun.fn()(self.__end_conversation)

    def __end_conversation(self, reason: str = None):
        """
        Mark the end of the conversation

        Parameters
        ----------
        reason : str, optional (default=None) The reason for ending the conversation
        """
        self.__end_of_conversation = True
        self.__end_conversation_reason = reason

    def run(self, task: str):
        console = Console()
        messages = [
            {
                "role": "system",
                "content": """
            You are a task assistant.
            When you are done, call the __end_of_conversation function with an optional message that tells the user why you are ending the conversation.
            """.strip(),
            },
            {"role": "user", "content": task},
        ]

        functions = self.taifun.functions_as_dict()

        while not self.__end_of_conversation:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=messages,
                functions=functions,
                function_call="auto",  # auto is default, but we'll be explicit
            )
            message_dict = response["choices"][0]["message"]
            response_message = Message(
                role=message_dict["role"],
                content=message_dict["content"] or "",
                function_call=FunctionCall(
                    name=message_dict["function_call"]["name"],
                    arguments=message_dict["function_call"]["arguments"],
                )
                if message_dict.get("function_call")
                else None,
            )

            messages.append(
                response_message.model_dump(exclude_none=True)
            )  # extend conversation with assistant's reply
            # Step 2: check if GPT wanted to call a function
            if response_message.function_call:
                console.print("Function call:", style="bold")
                console.print(response_message.function_call.name)

                function_name = response_message.function_call.name
                function_arguments = response_message.function_call.arguments
                function_response = self.taifun.handle_function_call(
                    {"name": function_name, "arguments": function_arguments}
                )

                messages.append(
                    Message(
                        role=Role.function,
                        name=function_name,
                        content=json.dumps(function_response),
                    ).model_dump(exclude_none=True)
                )

            elif response_message.content:
                console.print(response_message.role, style="bold")
                console.print(response_message.content)

            else:
                raise TaifunConversationRunnerException(
                    f"Unknown response: {response_message.content}"
                )

        return self.__end_conversation_reason

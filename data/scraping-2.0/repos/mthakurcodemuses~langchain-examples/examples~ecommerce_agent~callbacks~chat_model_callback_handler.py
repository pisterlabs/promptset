from langchain.callbacks.base import BaseCallbackHandler
from ecommerce_agent.utils.boxen_util import boxen_print


class ChatModelCallbackHandler(BaseCallbackHandler):

    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n\n\n===================== Sending Messages =====================\n\n\n\n")

        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")

            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(f"Running tool {call['name']} with args {call['arguments']}", title=message.type,
                            color="cyan")

            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")

            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="magenta")

            else:
                boxen_print(message.content, title=message.type, color="orange")

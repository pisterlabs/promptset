from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen 

# print(
#     boxen("YOUR TEXT", title="Human", color="yellow")
# )

def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))

# boxen_print("name is joshua abok", title="Josh", color="purple")


# extend the BaseCallbackHandler
class ChatModelStartHandler(BaseCallbackHandler):
    def on_chart_model_start(self, serialized, messages, **kwargs):
        print("\n\n\n\n============= sending messages =============\n\n")
        for message in messages[0]: 
            print(message.type)
            if message.type == "system": 
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human": 
                boxen_print(message.content, title=message.type, color="green")

            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}",
                    title=message.type, 
                    color="cyan"
                )

            elif message.type == "ai": 
                boxen_print(message.content, title=message.type, color="blue")

            elif message.type == "function": 
                boxen_print(message.content, title=message.type, color="purple")

            else: 
                boxen_print(message.content, title=message.type)

# print(ChatModelStartHandler.on_chart_model_start(serialized, messages, **kwargs))
import openai

from chat_backend import BaseBackend
from chat_log import MessageLog

class GPTTurboBackend(BaseBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        openai.api_key = self.settings.OPENAI_API_KEY

    def get_response(self, message_log: MessageLog, bot_name: str) -> str:
        """
        Get Response from GPT-3.5-Turbo
        """

        print("Contacting Python API for GPT-Turbo response...")
              
        # Grab the ChatML of the message log
        chat_ml = message_log.to_chat_ml(char_limit=10000) # 16000 is the max length of a message in ChatGPT apparently!
        raw = message_log.__str__()

        print(f"ChatML (LEN{len(raw)}):" + str(chat_ml))

        # Prepend the ChatML with the GPT-Turbo prompt
        chat_ml.insert(0, {"role": "assistant", "content": f"{bot_name}: Example message response."})
        chat_ml.insert(0, {"role": "user", "content": f"User: Example message query."})
        chat_ml.insert(0, {"role": "system", "content" : f"Your are an AI, {bot_name}. Both you and the user will respond with a name and a message separated by a colon (e.g. {bot_name}: Hello)."})

        # Send the ChatML to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=chat_ml
        )

        # Print the response
        print("Response:" + str(response))

        # Grab the response, trim the `{bot_name}:` at the start
        response_text = response["choices"][0]["message"]["content"]

        if response_text.startswith(f"{bot_name}:"):
            response_text = response_text[len(f"{bot_name}:"):].strip()

        # Return the response
        return response_text

    def get_summary(self, message_log: MessageLog, bot_name: str) -> str:
        """
        Get Response from GPT=3.5-Turbo
        """

        print("Contacting Python API for GPT-Turbo response...")
              
        # Grab the ChatML of the message log
        chat_ml = message_log.to_chat_ml()

        print("ChatML:" + str(chat_ml))

        # Add the prompt
        chat_ml.append({"role": "user", "content": f"What is an appropriate summary of this conversation in less than 10 words?"})

        # Send the ChatML to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=chat_ml
        )

        # Print the response
        print("Response:" + str(response))

        # Grab the response, trim the `{bot_name}:` at the start
        response_text = response["choices"][0]["message"]["content"]

        if response_text.startswith(f"{bot_name}:"):
            response_text = response_text[len(f"{bot_name}:"):].strip()

        # Return the response
        return response_text


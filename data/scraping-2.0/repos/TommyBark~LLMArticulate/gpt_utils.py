import os
import openai

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FIRST_PROMPT = """I am going to provide you with binary classifier inputs and outputs. The classifying rule is a simple short English sentence. This is not a sentiment classifier! 
Then I am going to just provide you with the inputs and you will have to predict the outputs. 
Please respond in the same format as is the format of examples, that is "INPUT: ... OUTPUT: label"\n\n"""

SECOND_PROMPT = """ Can you tell me rule you used to predict the outputs? It should be a simple one sentence description. 
                    Respond just with the rule itself."""


class GPTChat:
    def __init__(self, system_message: str = "", model: str = "gpt-4-1106-preview"):
        self.model = model
        self.messages = [
            {"role": "system", "content": system_message},
        ]
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY).chat

    def add_message(self, content: str, role: str = "user") -> None:
        self.messages.append({"role": role, "content": content})

    def send_message(self, message: str, role: str = "user") -> str:
        self.add_message(message, role)

        response = self.client.completions.create(
            model=self.model, messages=self.messages, max_tokens=1000
        )
        ai_text = response.choices[0].message.content
        if ai_text is None:
            ai_text = ""

        self.add_message(ai_text, "assistant")
        return ai_text


class AsyncGPTChat:
    def __init__(self, system_message: str = "", model: str = "gpt-4-1106-preview"):
        self.model = model
        self.messages = [
            {"role": "system", "content": system_message},
        ]
        self.client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY).chat

    def add_message(self, content: str, role: str = "user"):
        self.messages.append({"role": role, "content": content})

    async def send_message(self, message: str, role: str = "user") -> str:
        self.add_message(message, role)

        response = await self.client.completions.create(
            model=self.model, messages=self.messages, max_tokens=1000
        )
        ai_text = response.choices[0].message.content
        if ai_text is None:
            ai_text = ""

        self.add_message(ai_text, "assistant")
        return ai_text


async def manage_chat_async(chat_session, user_message):
    return await chat_session.send_message(user_message)

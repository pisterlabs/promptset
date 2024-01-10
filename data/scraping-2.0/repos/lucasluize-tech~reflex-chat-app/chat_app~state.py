import reflex as rx
import asyncio
import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class State(rx.State):
    question: str
    chat_history: list[tuple[str, str]]

    async def answer(self) -> str:
        # chatbot
        session = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.question},
            ],
            stop=None,
            temperature=0.7,
            stream=True,
        )

        # add to answer
        answer = ""
        self.chat_history.append((self.question, answer))
        for i in session:
            if hasattr(i.choices[0].delta, "content"):
                print(i.choices[0].delta.content)
                answer += i.choices[0].delta.content
                self.chat_history[-1] = (self.question, answer)
                yield

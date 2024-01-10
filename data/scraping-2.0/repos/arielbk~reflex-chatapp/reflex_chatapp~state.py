import os

import openai
import reflex as rx

openai.API_KEY = os.environ["OPENAI_API_KEY"]


class State(rx.State):
    # the current question being asked
    question: str

    # keep track of the chat history as a list of (question, answer) pairs
    chat_history: list[tuple[str, str]]

    async def answer(self):
        # our chatbot finally has some smarts
        session = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.question},
            ],
            stop=None,
            temperature=0.9,
            stream=True,
        )

        # stream the response back to the user
        answer = ""
        question = self.question
        self.question = ""

        self.chat_history.append((question, answer))
        for item in session:
            if hasattr(item.choices[0].delta, "content"):
                answer += item.choices[0].delta.content
                self.chat_history[-1] = (question, answer)
                yield

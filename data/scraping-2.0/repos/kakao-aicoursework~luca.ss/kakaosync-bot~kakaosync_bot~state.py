import reflex as rx
import os
import openai

from core.bot import OpenaiBot

openai.api_key = os.environ.get('OPENAI_API_KEY', None)


class State(rx.State):

    # The current question being asked.
    question: str

    # Keep track of the chat history as a list of (question, answer) tuples.
    chat_history: list[tuple[str, str]]

    def answer(self):
        # Our chatbot has some brains now!

        bot = OpenaiBot()
        session = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.question}
            ],
            stop=None,
            temperature=0.7,
            stream=True,
        )

        # Add to the answer as the chatbot responds.
        answer = ""
        self.chat_history.append((self.question, answer))

        # Clear the question input.
        self.question = ""
        # Yield here to clear the frontend input before continuing.
        yield

        answer = bot.ask(self.question)
        self.chat_history[-1] = (
            self.chat_history[-1][0],
            answer,
        )
        yield

        # for item in session:
        #     if hasattr(item, "content"):
        #         print(item.choices[0])
        #         answer += item.choices[0].delta.content
        #         self.chat_history[-1] = (
        #             self.chat_history[-1][0],
        #             answer,
        #         )
        #         yield
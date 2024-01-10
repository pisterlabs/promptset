from dotenv import load_dotenv
from openai import OpenAI
import os
from datetime import date

class Predictive:
    def __init__(self,question, contexts):
        self.question=question
        self.contexts=contexts
        self.messages=[]
    
    def _build_context(self):
        header_messages={"role": "system", "content": f"Eres un asistente de la universidad y tienes acceso a información sobre la universidad en la conversación actual. Utiliza ese conocimiento para responder a las preguntas. Actualmente estamos en el año {date.today().year}"}
        body_messages = [{"role": "assistant", "content": context} for context in self.contexts]
        question_messages = {"role": "user", "content": self.question}
        self.messages = [header_messages] + body_messages + [question_messages]
        return self.messages

    def get_answer_question(self):
        load_dotenv()
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self._build_context(),
            temperature=0
        )
        return response.choices[0].message.content
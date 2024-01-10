import langchain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt
from langchain.output_parsers import OutputFixingParser

# output_parser = DatetimeOutputParser()
#
# misformatted = result.content
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.cache import InMemoryCache
from langchain import PromptTemplate
import os
import openai
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import DatetimeOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"
api_key = "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"
llm = OpenAI()
# chat = ChatOpenAI(openai_api_key=api_key)


class HistoryQuiz:
    def create_history_question(self, topic):
        system_template = "You write single quiz questions about {topic}. You only return the quiz question."
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )

        human_template = "{question_request}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        q = "Give me a quiz question where the correct answer is a specific date."
        request = chat_prompt.format_prompt(
            topic=topic, question_request=q
        ).to_messages()

        chat = ChatOpenAI(openai_api_key=api_key)
        result = chat(request)

        return result.content

    def get_AI_answer(self, question):
        output_parser = DatetimeOutputParser()
        system_template = "You answer quiz questions with just a date."
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        human_template = """Answer the user's question:

               {question}

               {format_instructions}"""
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        format_instructions = output_parser.get_format_instructions()

        request = chat_prompt.format_prompt(
            question=question, format_instructions=format_instructions
        ).to_messages()
        chat = ChatOpenAI(openai_api_key=api_key)
        result = chat(request)
        correct_datetime = output_parser.parse(result.content)
        return correct_datetime


quiz_bot = HistoryQuiz()
question = quiz_bot.create_history_question(topic="World War 2")
print(question)

ai_answer = quiz_bot.get_AI_answer(question)
print(ai_answer)

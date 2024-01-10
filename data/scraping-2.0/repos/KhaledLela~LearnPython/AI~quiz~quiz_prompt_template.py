import json
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import (ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import SystemMessage


def language_structure():
    with open('lang.csv') as file:
        return file.read()


def quiz_structure():
    with open('quiz_structure.json') as file:
        return json.load(file)


def quiz_chat_prompt():
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that make Quiz."),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessage(
            content=f"Quiz has the following JSON structure {quiz_structure()}"),
        SystemMessagePromptTemplate.from_template(
            "Quiz has {question_count} multiple-choice questions, "
            "with each question having {alternative_count} options."),
        SystemMessagePromptTemplate.from_template(
            "Quiz aiming for {difficulty_level}"),
        HumanMessagePromptTemplate.from_template("""Write a quiz from the following:

        {text}

        QUIZ IN {language} ONE line double quotes JSON without explanation: """)
    ])


def quiz_pydantic_prompt(parser: PydanticOutputParser):
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Your helpful assistant for creating quizzes from concise summaries. tailored to {difficulty_level}"),
        SystemMessage(content=f"Language ids: {language_structure()}"),
        SystemMessagePromptTemplate.from_template(
            "Quiz question has {alternative_count} options."),
        SystemMessage(content=parser.get_format_instructions()),
        HumanMessagePromptTemplate.from_template("""Generate a multiple-choice {language} quiz with {question_count} questions from the following:

        {text}

        """)
    ])

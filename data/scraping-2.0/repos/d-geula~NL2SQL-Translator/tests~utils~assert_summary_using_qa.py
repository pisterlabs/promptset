import ast
from textwrap import dedent

import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)


def assert_summary_using_qa(summary: str, questions_and_answers: dict):
    chat = ChatOpenAI(temperature=0.0, verbose=True)  # type: ignore

    langchain.llm_cache = SQLiteCache(database_path="cache/.langchain.db")

    system_template = dedent("""
    This function extracts categories from a generated summary based on the questions provided.

    Args:
        summary (str): The generated summary.
        questions (list): A list of questions.

    Returns:
        dict: A dictionary of extracted categories. 
              Values represent single word categories where possible.

    Example:
        The returned dictionary may look something like this:

        "Question1": "Value1",
        "Question2": "Value2",
        ...""")

    human_template = dedent("""
    summary: {summary}
    questions: {questions}""")

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chat_response = chat(
        chat_prompt.format_prompt(
            summary=summary, questions=list(questions_and_answers.keys())
        ).to_messages()
    ).content
    
    parsed_response = ast.literal_eval(chat_response)

    for question, expected_answer in questions_and_answers.items():
        answer = parsed_response[question]
        assert answer == expected_answer

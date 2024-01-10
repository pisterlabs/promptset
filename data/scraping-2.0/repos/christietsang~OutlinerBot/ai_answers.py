"""
Generate answer using OpenAI API
"""

import openai
import ast
import os

# Insert API key here
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_ORGANIZATION = os.environ['OPENAI_ORGANIZATION']


def get_data(file_name: str) -> list[str, ...] or str:
    """
    Extract data from a file and return a list if the file contains a list.

    :param file_name: a string representing the name of the file
    :precondition: file_name must include the file extension
    :postcondition: if the file contains a list, this function returns a list object rather than
                  a string
    :return: a string with the data inside the file, or a list if the file contains a list
    """
    try:
        with open(file_name, errors="ignore") as text_file:
            data = text_file.read()
    except FileNotFoundError:
        print(f"\"{file_name}\" does not exist.", )
    else:
        if data[0] == "[" and data[-1] == "]":
            return ast.literal_eval(data)
        else:
            return data


def answer_question(documents: list[str, ...], question: str) -> str:
    """
    Answer a question using the OpenAI API.

    :param documents: a list of strings for the AI to search for the answer in
    :param question: a string containing the question for the AI
    :precondition: documents must be a list of strings
    :precondition: each string in documents must be less than 2000 tokens long
    :precondition: documents must hold 200 strings or less
    :return: a string with the AI's answer and accuracy
    """
    openai.organization = OPENAI_ORGANIZATION
    openai.api_key = OPENAI_API_KEY

    sample_outline = """Course Credits 5
  Minimum Passing Grade 50%
  Start Date January 04, 2022
  End Date April 22, 2022
  Total Hours 75
  Total Weeks 15
  Hours/Weeks 5
  Criteria % Comments
  Weekly quizzes 10 Short in-lab and in-class quizzes and coding activities
  Weekly labs 30 Weekly time-restricted programming exercises"""
    sample_questions = [[
        "What grade do I need to pass this course?", "A minimum of 50%"
    ], ["What is the start date of this class?", "January 04, 2020"],
        ["How much are the labs worth for this class?", "30%"]]

    result = openai.Answer.create(
        search_model="ada",
        model="curie",
        question=question,
        documents=documents,
        examples_context=sample_outline,
        examples=sample_questions,
        max_tokens=200,
        stop=["\n", "<|endoftext|>"],
    )

    answer = result["answers"][0]
    answer.replace("\n", "")
    score = result["selected_documents"][0]["score"]
    if score > 200:
        accuracy = "high"
    elif score > 100:
        accuracy = "medium"
    elif score > 0:
        accuracy = "low"
    else:
        accuracy = "negative"

    return f"{answer}\nAccuracy: {accuracy}"


def ask(question):
    """
    Drive the program

    :param question: a string containing the question for the AI
    :return: a string with the AI's answer and accuracy
    """
    documents = get_data("documents.txt")
    #  Drop any item that has more than 1000 tokens
    for index, item in enumerate(documents):
        if len(item.split()) > 1000:
            documents.pop(index)
    return answer_question(documents, question)

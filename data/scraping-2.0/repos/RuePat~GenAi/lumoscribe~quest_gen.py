import openai
import json

openai.api_key = ""

def get_completion(prompt, engine = 'text-davinci-003'):
    response = openai.Completion.create(
        engine = engine,
        prompt = prompt,
        max_tokens = 2500,
        n = 1  # Generate one response with 12 flashcards
    )

    # return json.loads(response.choices[0].text)
    return response.choices[0].text

def generate_mcq(text):
    prompt=f"""
        Create 5 Multiple Choice Questions, along with the options and correct answer based on the text delimited by triple backticks.\
        Provide them in a JSON format with the following keys in double quotes:
        question_text, options, correct_answer.
        question_text: Python string containing the text of the question
        options: Python list containing the strings of the options for the question
        correct_answer: Corresponding index of the answer to the question based on the 'options' list.
        ```{text}```
        """
    mcqs = get_completion(prompt)

    return mcqs

def generate_fib(text):
    prompt=f"""
        Create 5 Fill in the blanks Questions, along with the correct answer based on the text delimited by triple backticks.\
        Provide them in a JSON format with the following keys in double quotes:
        question_text, correct_answer.
        question_text: Python string containing the text of the question, where the blank/missing word is denoted by the token '<BLANK>'
        correct_answer: A Python list of 3 alternate options, any of which can appropriately fill the '<BLANK>' token from the 'question_text' string.
        ```{text}```
        """
    fibs = get_completion(prompt)

    return fibs

def generate_tf(text):
    prompt=f"""
        Create 5 True or False Statement, along with the correct statement (T/F) based on the text delimited by triple backticks.\
        Provide them in a JSON format with the following keys in double quotes:
        statement, sentiment.
        statment: Python string containing the text of the statement
        sentiment: A Python string (true / false) reflecting whether 'statement' is True or False.
        ```{text}```
        """
    tfs = get_completion(prompt)

    return tfs

def generate_flashcards(text):
    prompt = f"""
    Create 5 flashcards based on the text delimited by triple backticks.\
    Provide them in a JSON format with the following keys in double quotes:
    question, answer.
    question: Python string containing the text of the question
    answer: Python string containing the text of the answer
    ```{text}```
    """

    flashcards = get_completion(prompt)

    return flashcards

def generate_summary(text):
    prompt = f"""
    Summarize the following text delimited by triple backticks.\
    Provide the generated summary as a string
    ```{text}```
    """

    summary = get_completion(prompt)

    return summary

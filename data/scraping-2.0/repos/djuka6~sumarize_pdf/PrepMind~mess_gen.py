import os
import openai
from lingua import Language, LanguageDetectorBuilder


def process_message(input_message, type):
    API_KEY = os.environ["API_KEY"]
    openai.api_key = API_KEY
    languages = [
        Language.ENGLISH,
        Language.FRENCH,
        Language.GERMAN,
        Language.SPANISH,
        Language.AFRIKAANS,
        Language.ALBANIAN,
        Language.SERBIAN,
    ]
    detector = (
        LanguageDetectorBuilder.from_languages(*languages)
        .with_minimum_relative_distance(0.9)
        .build()
    )

    if type == "quiz":
        
        prompt = (
            "From this text: "
            + input_message
            + """. Generate a quiz with 5 questions, each question having 5 multiple-choice options. Note that there should be exactly 10 questions. The questions and options must make sense. The correct answer must be among the given options as an identical value. Give me the answer in the same language as the text itself. The answer should be in JSON format. Here is an example: [{"question": "value", "option1": "value", "option2": "value", etc..., "correct_answer": "value"}]. If the text provided doesn't allow for a meaningful quiz to be generated, simply return the word "null". If you generate a question, correct_answer must be one of the options. If the text contains English, Latin, or terms from any other language, the response must be in the language in which the text is originally written. """
        )
        prompt = "This is sumarized chanks of pdf of 30+ pages, sorted from 0(first page), to the n(last page). " + input_message + " Please ,  sumarize it and send me back the longest text that you can make out if this."
    elif type == "text":
       # prompt = (
          #  "From this text: "
          #  + input_message
          #  + ".Generate a general question in this language: "
          #  + " about this text suitable for exams. The text must make sense; otherwise, return '0'. IMPORTANT: Your response must be in the same language as this input text. "
       # )
       prompt = "Sumarize this text in English language but mention everything important : " + input_message

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    print(response)
    message_content = response.choices[0].message.content
    processed_message = message_content
    return processed_message


def process_message2(input_message, answer_value, question):
    print(type)
    API_KEY = os.environ["API_KEY"]
    openai.api_key = API_KEY

    prompt = (
        "This is the text: "
        + input_message
        + " This is the question: "
        + question
        + ". This is the answer: "
        + answer_value
        + "Rate an answer from 0 to 1. 0 is empty or non-related and 1 is correct and fully answer. Give your opinion what should be improved. "
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )
    print(response)
    message_content = response.choices[0].message.content
    processed_message = message_content
    return processed_message

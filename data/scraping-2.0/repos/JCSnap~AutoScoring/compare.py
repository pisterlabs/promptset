import os
from flask import Flask, request, jsonify
import openai
import constants
import requests
import json
import time


def format_answer(question, model_answer, student_answer):
    template = """
        Question: {}
        Model answer: {}
        Student answer: <> {} <>

        Format:
        Justification: [your justification]
        Score: [score]
    """.format(question, model_answer, student_answer)
    return template


def format_answer_without_justification(question, model_answer, student_answer):
    template = """
        Question: {}
        Model answer: {}
        Student answer: <> {} <>

        Format:
        Score: [score]
    """.format(question, model_answer, student_answer)
    return template


openai.api_key = os.getenv("OPENAI_API_KEY")

instruction_justification = "You are a teacher. You will be given a question, a model answer, and a student's answer. You are tasked to grade your student's answer out of 100 based on its accuracy when compared to the model answer. There can be some leeway and the answer need not follow the model answer word for word."
instruction = "You are a teacher. You will be given a question, a model answer, and a student's answer. You are tasked to grade your student's answer out of 100 based on its accuracy when compared to the model answer. There can be some leeway and the answer need not follow the model answer word for word. Do not provide explanation, just give the score."

models = ['gpt-3.5-turbo', 'gpt-4']

questions = constants.QUESTION
with open("results2.txt", "a") as file:
    for question in questions:
        file.write(f"Question: {question['question']}\n")
        file.write(f"Model Answer: {question['model_answer']}\n")
        file.write(f"Student Answer: {question['student_answer']}\n")
        for model in models:
            for instruction_type, format_fn in [(instruction_justification, format_answer), (instruction, format_answer_without_justification)]:
                prompt = format_fn(
                    question["question"], question["model_answer"], question["student_answer"])

                print(f"Sending request to {model} API...")
                start_time = time.time()
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": instruction_type},
                        {"role": "user", "content": prompt}
                    ]
                )
                text = completion.choices[0].message.content
                end_time = time.time()

                print(text)
                print(f"Time taken for {model}: {end_time - start_time}")

                # Extract the justification string if it is expected in the response
                justification = None
                if "Justification" in text:
                    start_index_j = text.find("Justification: ")
                    start_index_s = text.find("Score: ")
                    justification = text[start_index_j +
                                         len("Justification: "): start_index_s].strip()
                    print(f"{model} Justification: {justification}")

                # Extract the score value
                start_index_s = text.find("Score: ")
                score = text[start_index_s + len("Score: "):].strip()
                print(f"{model} Score: {score}")

                # Write the results to the file
                if "Justification" in text:
                    file.write(f"{model} Score (Justification): {score}\n")
                else:
                    file.write(f"{model} Score: {score}\n")

                file.write(
                    f"Time taken for {model}: {str(end_time - start_time)} seconds\n")

        file.write("=========================\n")

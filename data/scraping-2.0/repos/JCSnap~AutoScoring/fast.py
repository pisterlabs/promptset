import os
from flask import Flask, request, jsonify
import openai
import constants
import requests
import json
import time


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

instruction = "You are a teacher. You will be given a question, a model answer, and a student's answer. You are tasked to grade your student's answer out of 100 based on its accuracy when compared to the model answer. There can be some leeway and the answer need not follow the model answer word for word. Do not provide explanation, just give the score."

models = ['gpt-4']

ALL_QUESTIONS = [constants.QUESTIONS, constants.QUESTIONS2, constants.QUESTIONS_WRONG, constants.QUESTIONS_WRONG2, constants.QUESTIONS_CORRECT_BUT_LAYMAN, constants.QUESTIONS_CORRECT_BUT_GRAMMAR_BAD, constants.QUESTIONS_CORRECT_BUT_KEYWORD_PHRASING_DIFFERENT, constants.QUESTIONS_PARTIALLY_CORRECT, constants.QUESTIONS_CORRECT_BUT_INCORRECT_TERMS, constants.QUESTIONS_MISUNDERSTOOD_CONCEPTS,
                 constants.QUESTIONS_WRONG_BUT_CORRECT_TERMS, constants.QUESTIONS_CORRECT_BUT_WRONG_REASONING, constants.QUESTIONS_UNRELATED_IRRELEVANT, constants.QUESTIONS_WRONG_COMPLETELY, constants.QUESTIONS_NEGATIONS_AND_EXCEPTIONS, constants.QUESTIONS_CORRECT_BUT_DIFFERENT_CONTEXT, constants.QUESTIONS_SYNONYM, constants.QUESTIONS_GIBBERISH, constants.QUESTIONS_COMPLEX_STRCUTURES]
# questions = constants.QUESTION
for questions in ALL_QUESTIONS:
    with open("results2.txt", "a") as file:
        for question in questions:
            prompt = format_answer_without_justification(
                question["question"], question["model_answer"], question["student_answer"])
            print(f"Sending request to GPT API...")
            start_time_openai = time.time()
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]
            )
            text = completion.choices[0].message.content
            end_time_openai = time.time()

            print(text)
            print(f"Time taken: {end_time_openai - start_time_openai}")

            # Find the index of "Score: "
            start_index_s = text.find("Score: ")

            # Extract the score value
            score = text[start_index_s + len("Score: "):].strip()

            print(f"Openai Score: {score}")

            url = "https://gtfpdj13l7.execute-api.us-east-1.amazonaws.com/dev/qa"

            payload = {
                "data": [
                    question["model_answer"],
                    question["student_answer"],
                ]
            }

            json_payload = json.dumps(payload)

            headers = {
                "Content-Type": "application/json"
            }

            print("Sending request to Original API...")
            start_time_original = time.time()
            response = requests.post(url, data=json_payload, headers=headers)
            end_time_original = time.time()

            if response.status_code == 200:
                print("POST request successful!")
                scores = response.text
                # Parse the JSON data
                data = json.loads(scores)

                # Access the overall_score value
                overall_score = data['overall_score'][0]

                # Print the overall score
                print(f"Original Score: {overall_score}")
                print(
                    f"Time taken for Original API: {end_time_original - start_time_original} seconds")
            else:
                print(
                    f"POST request failed with status code: {response.status_code}")

            # Write the results to the file
            file.write("Question: " + question["question"] + "\n")
            file.write("Model Answer: " + question["model_answer"] + "\n")
            file.write("Student Answer: " + question["student_answer"] + "\n")
            file.write("OpenAI Score: " + score + "\n")
            file.write("Original Score: " + str(overall_score * 100) + "\n")
            file.write("Time taken for OpenAI API: " +
                       str(end_time_openai - start_time_openai) + " seconds\n")
            file.write("Time taken for Original API: " +
                       str(end_time_original - start_time_original) + " seconds\n\n")

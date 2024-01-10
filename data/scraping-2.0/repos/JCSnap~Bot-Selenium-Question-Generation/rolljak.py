import os
import os.path
from flask import Flask, request, jsonify
import openai
import constants
import re
import constants
import pandas as pd

openai.api_key = os.getenv("OPENAI_API_KEY")
file_path = "~/Downloads/excel.xlsx"
MODEL = "gpt-4"

# If the file exists, read the existing data, otherwise create an empty DataFrame
df = pd.read_excel(file_path)

number_of_questions = constants.number_of_questions
grade = constants.grade

QUESTION_SPLIT_INSTRUCTION = constants.prompt_check_instruction(
    constants.number_of_questions, constants.grade)

SETTINGS = """
Language: English
Student educational level: {grade} 
Number of questions: 1
Question type(s): MCQ
"""

topic = constants.topic
grade = constants.grade
subject = constants.subject
generator = "Rolljak"
type_of_question = "Multiple Choice"
number_of_correct_answer = "1"

# QUESTION SPLIT
completion = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": QUESTION_SPLIT_INSTRUCTION},
        {"role": "user", "content": constants.topic}
    ]
)

subtopics = completion.choices[0].message.content

lines = subtopics.split("\n")
print(lines)

subtopics = [re.sub(r'^\d+\. ', '', line)
             for line in lines if re.match(r'^\d+\. ', line)]

prompt = SETTINGS + "\n"

for i in range(len(subtopics)):
    try:
        prompt = SETTINGS + "\n" + subtopics[i] + constants.MCQ_FORMAT + "\n"
        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": constants.QUESTION_GENERATION_INSTRUCTION},
                {"role": "user", "content": prompt}
            ]
        )
        question = completion.choices[0].message.content
        print(question)

        subtopic_start = question.index("Subtopic:")
        question_start = question.index("Question:")
        options_start = question.index("Options:")
        answer_start = question.index("Answer:")

        # Parse the subtopic
        subtopic = question[subtopic_start +
                            len("Subtopic:"): question_start].strip()

        # Parse the question
        question_mcq = question[question_start +
                                len("Question:"): options_start].strip()

        # Parse the options
        options_text = question[options_start +
                                len("Options:"): answer_start].strip()
        options = re.findall(r'\. (.*)', options_text)

        # Parse the answer
        answer_text = question[answer_start:].strip()
        print(question_mcq)
        print(options_text)
        print(answer_text)
        match = re.search(r'Answer: (\w)', answer_text)
        answer_letter = match.group(1)
        answer = ord(answer_letter) - ord('A')

        # Get the correct option as a string and the incorrect options
        correct_option = options[answer]
        incorrect_options = options[:answer] + options[answer+1:]

        # Create a new row for this question
        new_row = {
            'Topic': topic,
            'Grade': grade,
            'Subject': subject,
            'Generator': generator,
            "Type of question": type_of_question,
            'Question': question_mcq,
            'Answer': number_of_correct_answer,
            'Content': correct_option,
            'Unnamed: 4': incorrect_options[0],
            'Unnamed: 5': incorrect_options[1],
            'Unnamed: 6': incorrect_options[2]
        }

        # Append the new row to the DataFrame
        df = df.append(new_row, ignore_index=True)
    except:
        print("Error")
        continue

# Write the entire DataFrame to excel
df.to_excel(file_path, index=False)

import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def create_test_prompt(topic, num_questions, num_possible_answers):
    prompt = f"Create a multiple choice quiz on the topic about {topic}. "
    prompt += f"with {num_questions} questions that starts with 'Q#: ' that is based on a fact and not opinionated. "
    prompt += f"Each question should have {num_possible_answers} possible answers starts with capital letter bullet points with this format 'A. '. "
    prompt += f"Also include the correct answer for each question with a starting string of 'Correct answer: '"
    return prompt

topic = input("What topic do you want to create a quiz on? ")
num_questions = 2

response = openai.Completion.create(engine="text-davinci-003", prompt=create_test_prompt(topic, num_questions, 4), max_tokens=100, temperature=0.7)

quiz = response['choices'][0]['text']

def create_student_view(quiz, num_questions):
  student_view = {1: ''}
  question_number = 1

  for line in quiz.split('\n'):
    if not line.startswith('Correct answer:'):
        student_view[question_number] += line + '\n'
    else:
        if question_number < num_questions:
            question_number += 1
            student_view[question_number] = ''

  return student_view

def create_answers_view(quiz, num_questions):
  answer_view = {1: ''}
  question_number = 1

  for line in quiz.split('\n'):
    if line.startswith('Correct answer:'):
        answer_view[question_number] += line + '\n'
        if question_number < num_questions:
            question_number += 1
            answer_view[question_number] = ''

  return answer_view

student_questions = create_student_view(quiz, num_questions)
teacher_answers = create_answers_view(quiz, num_questions)

def take_exam(student_questions):
  student_answers = {}
  for question_number, question_view in student_questions.items():
    print(question_view)
    answer = input("Answer: ")
    student_answers[question_number] = answer

  return student_answers

student_answers = take_exam(student_questions)

print(student_answers)

def grade_exam(student_answers, teacher_answers):
  score = 0
  for question_number, answer in student_answers.items():
    if answer == teacher_answers[question_number][16]:
        score += 1

  return score

scores = grade_exam(student_answers, teacher_answers)

print(f"\nYou got {scores} out of {num_questions} correct.")
print(f"\nYour score is {scores / num_questions * 100}%")

if (scores / num_questions * 100) > 70:
  print("\nYou passed the exam!")
else:
  print("\nYou failed the exam.")

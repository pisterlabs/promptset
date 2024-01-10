#  function for testing quiz generation 

import openai
import sys
sys.path.append('/Users/busterblackledge/')
from keys import openai_API_key

openai.api_key = openai_API_key

def answerCorrectGen(question):
    """generate the answers to questions"""

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
                {"role": "system", "content": f"give a very simple and correct answer to the following question: {question}"}
        ]
    )
    answer = response["choices"][0]["message"]["content"]
    return answer.replace("\n","")

def answerIncorrectGen(question):
    """generate some incorrect answers to the question"""

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
                {"role": "system", "content": f"Generate 4 very simple, but plausible, incorrect answer to the following question: {question}"}
        ]
    )
    answers = response["choices"][0]["message"]["content"]
    for answer in answers:
        answer.replace("\n","")

    return answers

def questionGen(script):
    """generate some questions based on the lecture script"""

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
                {"role": "system", "content": f"Generate a single question based upon the following text from a lecture that can be answered in a few words: {script}"}
        ]
    )
    # remove \n
    answer = response["choices"][0]["message"]["content"]
    return answer.replace("\n","")

def answerExplinationGeneration(answer,question):
    """generate an explination to the answer of a given question"""

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
                {"role": "system", "content": f"For this question: {question} very simply explain why the following answer is correct: {answer}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# what is the script
script = "Arsenal Football Club is a professional football club based in London, England. The club was founded in 1886 as Dial Square Football Club and in 1893 became known as Arsenal. They play their home matches at the Emirates Stadium and are one of the most successful clubs in English football history, having won 13 league titles and 14 FA Cups. Arsenal has a loyal fan base and is known for playing attacking and stylish football."
# generate the question
question = questionGen(script)
# generate the answer
correctAnswer = answerCorrectGen(question)
incorrectAnswer = answerIncorrectGen(question)
# generate the answer explaintion
explination = answerExplinationGeneration(correctAnswer, question)

quiz = {
    "question": question,
    "correct answer": correctAnswer,
    "incorrect answer": incorrectAnswer,
    "explination": explination,
}

print(quiz)

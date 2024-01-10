import openai
import random


def Get_Question(arr):
    reference = random.randint(1, len(arr)-2)
    package = {"reference": reference, "question": ''}
    paragraph = arr[reference]
    question_type = random.randint(1, 2)

    if question_type == 1:
        package["question"] = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"You have taught the user {paragraph}. Now ask them a challenging hard multiple choice question that has only one "
                   f"correct option without giving them the answer",
            temperature=0.5,
            max_tokens=300,
            frequency_penalty=0.2,
            presence_penalty=0.0
        )["choices"][0]["text"].strip()

    else:
        package["question"] = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"You have taught the user {paragraph}. Now ask them a challenging hard True or False question without giving them "
                   f"the answer",
            temperature=0.5,
            max_tokens=300,
            frequency_penalty=0.2,
            presence_penalty=0.0
        )["choices"][0]["text"].strip()

    question = package["question"]
    package["answer"] = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Give answer to the following question and explain your solution with detail {question}",
        temperature=0.25,
        max_tokens=400,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )["choices"][0]["text"].strip()

    return package

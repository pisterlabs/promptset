import os
import openai


def generateAnswer(userQuestion):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=userQuestion,
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
        best_of=1,
        frequency_penalty=0.47,
        presence_penalty=0.31,
    )
    answer = response["choices"][0]["text"]
    return answer

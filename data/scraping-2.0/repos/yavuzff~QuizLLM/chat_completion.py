import openai
import json
import os

history = [{"role": "system",
            "content": "You are a quiz generator which when given a topic, returns a question with 4 options, 1 correct answer and an explanation in JSON format. Do not ask the same/similar questions to what has already been asked. Do not use options returned in previous questions and make the questions gradually harder."},
           {"role": "user", "content": "Premier League"},
           {"role": "assistant",
            "content": '{"question": "Who is the all-time top scorer in the Premier League?", "options": ["Harry Kane", "Wayne Rooney", "Alan Shearer", "Michael Owen"], "answer": "Alan Shearer", "explanation":"Shearer has scored 260 goals whereas Kane, Rooney and Owen have scored 213, 208 and 150 respectively.}'}
           ]

def get_question_from_topic(topic, api_key=None):
    global history

    if api_key is not None:
        openai.api_key = api_key
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    message = [{"role": "user", "content": topic}]
    history = history + message
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history)
    print("API response:\n", response)

    quiz_response = response.choices[0].message.content
    # sample quiz response value:
    # {"question": "What is the name of the school that Harry Potter attends?", "options": ["Durmstrang Institute", "Beauxbatons Academy of Magic", "Hogwarts School of Witchcraft and Wizardry", "Ilvermorny School of Witchcraft and Wizardry"], "answer": "Hogwarts School of Witchcraft and Wizardry", "explanation": "Hogwarts is the primary setting for the Harry Potter series. Durmstrang Institute, Beauxbatons Academy of Magic, and Ilvermorny School of Witchcraft and Wizardry are other wizarding schools in the Harry Potter universe."}

    history = history + [{"role": "assistant", "content": quiz_response}]

    try:
        return json.loads(quiz_response)
    except ValueError as e:
        raise Exception(
            f"Cannot convert OpenAI response to JSON. This was the OpenAI response:\n {quiz_response}\n") from e


if __name__ == '__main__':
    # uses environment variable or set below
    # openai.api_key = api_key

    # models = openai.Model.list()
    # print(models)

    result = get_question_from_topic("Harry Potter")
    print(result)

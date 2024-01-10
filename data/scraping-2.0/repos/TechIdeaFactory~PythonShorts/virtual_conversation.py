# pip install openai
# pip install --upgrade openai

import openai

openai.api_key = "your key"


def call_chatgpt(
    system_message: str,
    user_message: str,
) -> str:

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    generated_text = response[
        "choices"
    ][0]["message"]["content"]

    return generated_text


def answer_and_question(
    ask_person: str,
    reply_person: str,
    question: str,
) -> (str, str):

    system_message = (
        f"You are {reply_person}."
    )
    user_message = (
        f" Answer the following question "
        f" in 2 sentences. {question} ?"
        f" The answer must be contained inside "
        f" square brackets. The answer"
        f" must be inside square brackets. "
        f" Then ask a question "
        f" to {ask_person}  using"
        f" two sentences. The question"
        f" must be contained inside "
        f" curly brackets. "
        f" The question must be inside"
        f" curly brackets. "
        f" The question"
        f" must not include the name of"
        f" the person to which it "
        f" is addressed. The question "
        f" must be unexpected and not"
        f" an obvious question. The conversation"
        f" must surprising and reveal deeper"
        f" aspects of personality that very "
        f" few people know. The conversation must"
        f"be surprising and educational and"
        f"factually correct."
    )

    response = call_chatgpt(
        system_message, user_message
    )
    answer = response.split("[", 1)[
        1
    ].split("]")[0]
    question = response.split("{", 1)[
        1
    ].split("}")[0]
    return (answer, question)


def chat(iter_num: int, person1: str,
                person1_descr: str, person2: str,
                person2_descr: str,
                question: str):

    person1_identity = f" {person1} {person1_descr}"
    person2_identity = f" {person2} {person2_descr}"

    for x in range(iter_num):

        print(
            f"[{person1}] {person2} "
            + question
        )

        (
            answer,
            question,
        ) = answer_and_question(
            person1_identity,
            person2_identity,
            question,
        )

        print(f"[{person2}] " + answer)
        print(
            f"[{person2}] {person1} "
            + question
        )

        (
            answer,
            question,
        ) = answer_and_question(
            person2_identity,
            person1_identity,
            question,
        )

        print(f"[{person1}] " + answer)


chat(2, "Grace Hopper",
     "the famous computer scientist",
     "Dennis Ritchie",
     "the famous computer scientist ",
     " what makes programming languages powerful? ")

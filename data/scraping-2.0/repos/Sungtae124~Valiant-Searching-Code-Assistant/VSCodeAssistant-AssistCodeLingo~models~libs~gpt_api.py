from openai import OpenAI

KEY = ''

def get_questions(words):

    client = OpenAI(api_key=KEY)

    question = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Create three natural questions by seamlessly combining some of the given words."},
            {"role": "user", "content": ', '.join(words)},
        ],
    )

    return question.choices[0].message.content


def get_response(question):

    client = OpenAI(api_key=KEY)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "You are a python tutor."},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content
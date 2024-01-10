import openai
import time as t


def get_short_description(description):
    openai.api_key_path = 'path_api_key'

    question = "Shorten this game description to 3-4 sentences removing all html markups (<br> etc.)\n\n"

    start = t.time_ns()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": question + description
             }
        ]
    )

    stop = t.time_ns()

    print((stop - start) / 10 ** 6)
    print(description,response['choices'][0]['message']['content'])


    return response['choices'][0]['message']['content']


def get_category_pl(category):
    openai.api_key_path = 'path_api_key'

    question = "Translate this category to Polish\n\n"

    start = t.time_ns()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": question + category
             }
        ]
    )

    stop = t.time_ns()

    print((stop - start) / 10 ** 6)
    print(category,response['choices'][0]['message']['content'])


    return response['choices'][0]['message']['content']


def get_short_description_pl(description):
    openai.api_key_path = 'path_api_key'

    question = "Translate this description to Polish\n\n"

    start = t.time_ns()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": question + description
             }
        ]
    )

    stop = t.time_ns()

    print((stop - start) / 10 ** 6)
    print(description,response['choices'][0]['message']['content'])


    return response['choices'][0]['message']['content']

def get_names_pl(name):
    openai.api_key_path = 'path_api_key'

    question = "Translate this name to Polish\n\n"

    start = t.time_ns()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user",
             "content": question + name
             }
        ]
    )

    stop = t.time_ns()

    print((stop - start) / 10 ** 6)
    print(name,response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']
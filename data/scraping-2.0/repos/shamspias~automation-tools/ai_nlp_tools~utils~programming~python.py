from django.conf import settings
import openai

openai.api_key = settings.OPEN_AI_KEY


def one_line_list_comprehension(prmt):
    """
    :param prmt:
    :return: context dist
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="create one-line list comprehension: \n\n{}\n".format(prmt),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text']
    context = {
        'data': [my_text]
    }
    return context


def one_line_dist_comprehension(prmt):
    """
    :param prmt:
    :return: context dist
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="create one-line dictionary comprehension: \n\n{}\n".format(prmt),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text']
    context = {
        'data': [my_text]
    }
    return context


def one_line_generator(prmt):
    """
    :param prmt:
    :return: context dist
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="create one-line generator: \n\n{}\n".format(prmt),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text']
    context = {
        'data': [my_text]
    }
    return context

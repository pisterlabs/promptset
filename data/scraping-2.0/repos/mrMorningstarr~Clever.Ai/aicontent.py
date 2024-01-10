import os
import openai
import config
openai.api_key = config.OPENAI_API_KEY


def productDescription(query):
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt="Generate detailed product description for: {}".format(query),
        temperature=0.5,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']

        else:
             answer = "Oops Sorry, You baet AI this time"

    else:
        answer= 'Oops Sorry, You beat AI this time'

    return answer

def jobDescription(query):
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt="Generate detailed and professional job description for: {}".format(query),
        temperature=0.5,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']

        else:
             answer = "Oops Sorry, You baet AI this time"

    else:
        answer= 'Oops Sorry, You beat AI this time'

    return answer

def tweetidea(query):
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt="Generate professional and witty tweet for given: {}".format(query),
        temperature=0.5,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']

        else:
             answer = "Oops Sorry, You baet AI this time"

    else:
        answer= 'Oops Sorry, You beat AI this time'

    return answer


def coldEmail(query):
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt="Generate professional and witty cold email for the following: {}".format(query),
        temperature=0.5,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']

        else:
             answer = "Oops Sorry, You baet AI this time"

    else:
        answer= 'Oops Sorry, You beat AI this time'

    return answer

def coldEmail(query):
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt="Generate professional and witty cold email for the following: {}".format(query),
        temperature=0.5,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if 'choices' in response:
        if len(response['choices']) > 0:
            answer = response['choices'][0]['text']

        else:
             answer = "Oops Sorry, You baet AI this time"

    else:
        answer= 'Oops Sorry, You beat AI this time'

    return answer



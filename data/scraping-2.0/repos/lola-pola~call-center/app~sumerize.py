import openai
import os




def find_keywords(text,model_name):
    response = openai.Completion.create(
        engine=model_name,
        prompt="Extract keywords from: " + text,
        max_tokens=1024,
        n=1,
        temperature=0.5,
        stop=None

    )

    keywords = response.choices[0].text
    return keywords.strip().split("\n")


def find_something(text,something,model_name):
    response = openai.Completion.create(
        engine=model_name,
        prompt=something + text,
        max_tokens=1024,
        n=1,
        temperature=0.5,
        stop=None

    )

    keywords = response.choices[0].text
    return keywords.strip().split("\n")

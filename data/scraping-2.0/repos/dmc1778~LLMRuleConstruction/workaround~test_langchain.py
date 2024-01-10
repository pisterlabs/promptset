from collections import Counter
from nltk.util import ngrams
import nltk
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from json.decoder import JSONDecodeError
import os
import openai
import json
from dotenv import load_dotenv
load_dotenv()

openai.organization = os.getenv("ORG_ID")
openai.api_key = os.getenv("API_KEY")


def driver():
    a = f"""10"""
    b = f"""20"""

    template_string = """
    Add ```{a}``` and ```{b}``` together.
    """
    # _parition_response_level1 = gpt_conversation(_prompt)
    # print(_parition_response_level1.choices[0].message.content)

    chat_ = ChatOpenAI(temperature=0.0, openai_api_key=os.getenv("API_KEY"))

    _prompt_template = ChatPromptTemplate.from_template(template_string)

    _message = _prompt_template.format_messages(
        a=a,
        b=b
    )

    response_ = chat_(_message)
    print(response_.content)


if __name__ == '__main__':
    # Example usage
    sentences = [
        "The cat is on the mat.",
        "The dog is playing in the park.",
        "The cat and the dog are friends."
    ]

    frequent_patterns = extract_frequent_patterns(
        sentences, ngram_size=2, min_frequency=2)

    # Print the frequent patterns
    for pattern in frequent_patterns:
        print(' '.join(pattern))

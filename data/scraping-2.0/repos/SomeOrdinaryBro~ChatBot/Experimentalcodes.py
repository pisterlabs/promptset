import re
import random
import requests
import spacy
import openai
from nltk.stem import WordNetLemmatizer
from spacy.errors import Errors

openai.api_key = "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"

MODEL_PROMPT = "Hello, I'm Jarvis. How can I assist you today?"

nlp = spacy.load("en_core_web_sm")
lem = WordNetLemmatizer()

GREETING_KEYWORDS = ("hello", "hi", "hey")
GREETING_RESPONSES = ["Hi there!", "Hello!", "Greetings!"]


def is_greeting(sentence):
    """Check if the sentence is a greeting."""
    for word in sentence:
        if word.lower() in GREETING_KEYWORDS:
            return True
    return False


def generate_response(user_input, temperature=0.7):
    prompt = f"{MODEL_PROMPT}\n{user_input}\n"
    doc = nlp(prompt)
    tokens = [lem.lemmatize(token.text) for token in doc]
    processed_prompt = " ".join(tokens)

    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=processed_prompt,
            temperature=temperature,
            max_tokens=2048,
            n=3,
            stop="END\n",
            timeout=60,
        )
        message = response.choices[0].text if response.choices else ""
        message = re.sub(r'[^\x00-\x7F]+', "", message)
        message = message.strip()
    except (requests.exceptions.RequestException, 
            openai.error.InvalidRequestError, Errors.DocError) as error:
        message = "Sorry, there was a problem generating a response. Please try again later."
        print(error)

    return message


def main():
    user_input = ""
    while not is_greeting(user_input):
        user_input = input("Hello, how can I assist you today? ")

    response = GREETING_RESPONSES[random.randint(0, 2)]
    print(response)

    print(MODEL_PROMPT)

    while True:
        try:
            user_input = input("> ")
            response = generate_response(user_input, temperature=0.7)
            print(response)
        except (requests.exceptions.RequestException, 
                openai.error.InvalidRequestError, Errors.DocError) as error:
            print("Sorry, there was a problem generating a response. Please try again later.")
            print(error)
            break
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()

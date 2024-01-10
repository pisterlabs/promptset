import openai
import requests

import textwrap
from fpdf import FPDF
import re
import time

API_KEY = "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"
model_engine = "text-davinci-003"


def main():
    # Set the API key and model
    openai.api_key = API_KEY
    temperature = 1

    # Set the prompt and temperature

    # Get the generated text
    # generated_text = response["choices"][0]["text"]

    subjects = open('Colleges/UF/classes.txt', 'r')

    current_subject = ""
    list_of_topics = []
    current_topic = ""

    for line in subjects.readlines():
        time.sleep(5)
        current_subject = repr(line).replace("\\n", "").replace("'", "")
        list_of_topics = topic_generator(current_subject)
        print(current_subject + ":" + str(list_of_topics))

        #print(list_of_topics)

    # Print the generated text
    # print(generated_text)


def topic_generator(subject):
    # Set the API key and model
    openai.api_key = API_KEY

    prompt = "Create a list of class 50 unique topics (1-3 words long) commonly found in a college textbook table of contents for " + subject + " course in a bulleted format. Do not duplicate any topics and make sure they are not similar to each other."

    # Make the request to the API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        temperature=0,
        max_tokens=1200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    # Get the generated text
    generated_text = response["choices"][0]["text"]

    topic_pattern = re.compile(("(.)\s?(.+)"))
    topics_found = topic_pattern.finditer(generated_text)
    topics = []
    for topic in topics_found:
        topics.append(topic.group(2).strip())

    return topics

if __name__ == "__main__":
    main()
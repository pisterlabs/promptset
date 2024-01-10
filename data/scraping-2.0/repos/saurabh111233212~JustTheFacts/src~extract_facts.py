import openai
import os
import re
import json


def load_json(filename):
  if os.path.isfile(filename):
    with open(filename) as json_file:
      data = json.load(json_file)
      return data
  return {}


def save_json(data, filename):
  with open(filename, "w") as outfile:
    json.dump(data, outfile)


def extract_statements(text):
    """ Extract statements from a numbered list, skipping any lines that don't start with a number"""
    pattern = r'^\s*\d+\.\s*(.*)\s*$'
    lines = text.split('\n')

    statements = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            statements.append(match.group(1))
    return statements


def ask_question_about_article(article_text, question):
    prompt_messages = [
        {
            "role": "system",
            "content": "You are a fact checker."  # The system description tells GPT what persona it should take on
        },
        {
            "role": "user",
            "content": "Here is a newspaper article:"
        },
        {
            "role": "user",
            "content": article_text
        },
        {
            "role": "user",
            "content": question
        },
    ]

    response = openai.ChatCompletion.create(model="gpt-4", messages=prompt_messages)

    return response["choices"][0]["message"]["content"]


def ask_question(question):
    prompt_messages = [
        {
            "role": "user",
            "content": question
        },
    ]

    response = openai.ChatCompletion.create(model="gpt-4", messages=prompt_messages)

    return response["choices"][0]["message"]["content"]

def extract_facts_from_article(article_text_1):
    question = "Please extract a list of simple, declarative sentences that enumerate just the facts from the article.   When you extract the facts, please rephrase the sentences that you extract them from."

    response = ask_question_about_article(article_text_1, question)
    return response


def fact_set_to_str(fact_sets):
    fact_set_str = ""
    for i in range(len(fact_sets)):
        fact_set_str += "Set {set_num}: {set_contents}\n".format(set_num=i, set_contents=str(fact_sets[i]))
    return fact_set_str


def get_closest_fact_set(fact_sets, fact):
    prompt_messages = [
        {
            "role": "system",
            "content": "You help group statement into sets of equivalent facts."  # The system description tells GPT what persona it should take on
        },
        {
            "role": "user",
            "content": "Here are the fact sets:"
        },
        {
            "role": "user",
            "content": fact_set_to_str(fact_sets)
        },
        {
            "role": "user",
            "content": "Here's a fact to assign to a fact set: " + fact
        },
        {
            "role": "user",
            "content": "Which fact set does it most closely correspond to?  "
                       "Give the number, or say 'None' if it doesn't closely correspond to any."
        },
    ]

    response = openai.ChatCompletion.create(model="gpt-4", messages=prompt_messages)

    return response["choices"][0]["message"]["content"]


def get_facts(url):
    import requests
    res = requests.get("http://just-the-facts.apps.allenai.org/api/get-facts", params={"url": url, "method": "gpt-4"})
    print(res.text)


def ask_openai_question():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    q = "Can you write the code of a browser plugin that takes as input 2 lists of strings and display them side by side?"
    response = ask_question(q)
    print(response)


if __name__ == "__main__":
    #get_facts(url='https://www.breitbart.com/news/wind-whipped-hawaii-wildfires-force-evacuations-water-rescues/')
    get_facts(url='https://www.nbcnews.com/news/world/americans-imprisoned-iran-prisoner-exchange-deal-rcna99105')


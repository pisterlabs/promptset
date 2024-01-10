import requests
from bs4 import BeautifulSoup 
import openai
import os

with open('key.txt','r') as f:
        api_key = f.read().strip('\n')
        assert api_key.startswith('sk-'), "Error loading your OpenAI API key from key.txt."

openai.api_key = api_key

url = 'https://en.wikipedia.org/wiki/Apple_Inc.'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

def getName(url):
    company_name = soup.find('h1', id='firstHeading')
    company_name = company_name.text

    return company_name

def getDescription(url):
    container = soup.find('div',id='bodyContent')
    box = container.find('div', id='mw-content-text')
    contents = box.find("div", class_="mw-parser-output")

    return contents

def getShortDescr(contents):
    short_descr = contents.find("div")
    short_descr = short_descr.text

    return short_descr 

def getLongDescr(contents):
    paragraphs = contents.find_all("p")
    long_descr = paragraphs[1].text

    return long_descr

def create_sdescr(company_name, long_descr):
    prompt = f"Reword the following input paragraph into a ONE SENTENCE company description, about the company, {company_name}:" \
             f"Focus on paraphrasing, using synonyms, and restructuring sentences to create a new, unique description.\n\n" \
             f"Be sure to retain the key information about the company, its products/services, and its target markets.\n" \
             f"Original Description:\n" \
             f"[{long_descr}]\n"


    messages = [
        {'role':'system','content':'Answer as concisely as possible.'},
        {'role':'user', 'content': prompt}
    ]

    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages,
        temperature = 0.8, 
        top_p = 1, 
        max_tokens = 500, 
        n = 1, 
        frequency_penalty = 1, 
        presence_penalty = 1, 
    )

    return response['choices'][0].message.content

def create_ldescr(company_name, long_descr):
    prompt = f"Using the following paragraph about the company, {company_name}, reword the description while maintaining the overall meaning and context. " \
             f"Focus on paraphrasing, using synonyms, and restructuring sentences to create a new, unique description.\n\n" \
             f"Be sure to retain the key information about the company, its products/services, and its target markets.\n" \
             f"Original Description:\n" \
             f"[{long_descr}]\n"


    messages = [
        {'role':'system','content':'Answer as concisely as possible.'},
        {'role':'user', 'content': prompt}
    ]

    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = messages,
        temperature = 0.8, 
        top_p = 1, 
        max_tokens = 1000, 
        n = 1, 
        frequency_penalty = 1, 
        presence_penalty = 1, 
    )

    return response['choices'][0].message.content

company_name = getName(url)
contents = getDescription(url)
short_descr = getShortDescr(contents)
long_descr = getLongDescr(contents)

final_sdescr = create_sdescr(company_name, long_descr)
final_ldescr = create_ldescr(company_name, long_descr)

#print("short descr: "+ short_descr)
#print("long descr: "+ long_descr)

print("\n")
print("final sdescr: "+ final_sdescr)
print("\n")
print("final ldescr: "+ final_ldescr)

#print(company_name)
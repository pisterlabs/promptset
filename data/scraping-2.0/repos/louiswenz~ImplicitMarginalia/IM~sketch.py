import scholarly
import requests
from bs4 import BeautifulSoup
import openai
import string
import pandas as pd
import re

# Set up OpenAI API key
# Replace with your OpenAI API key
openai.api_key = "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"


def strip_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    stripped_text = text.translate(translator)
    return stripped_text


def find_citing_articles(citation_sentence):
    citation_sentence = strip_punctuation(citation_sentence)
    search_url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={citation_sentence}"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles_title = []
    articles_context = []

    results = soup.find_all('div', class_='gs_ri')

    # print(response.text)

    for result in results:
        title_element = result.find('h3', class_='gs_rt')
        if title_element:
            title = title_element.text.strip()
            title = title.replace("[HTML]", "").strip()

            # Extracting the citation sentence and its context
            context_paragraph = result.find('div', class_='gs_rs').text
            # print(context_paragraph)
            context_paragraph = context_paragraph.replace("\n", "").strip()
            # Checking if the citation sentence is present in the context sentences
            if citation_sentence.lower() in context_paragraph.lower():
                articles_title.append(title)
                articles_context.append(context_paragraph)

    return articles_title, articles_context


def search_citing_paragraphs(sentence):
    sentence = strip_punctuation(sentence)
    # search_url = f"https://scholar.google.com/scholar?q={sentence}&hl=en&as_sdt=0,5"
    search_url = f"https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={sentence}"
    response = requests.get(search_url)
    page_content = response.text

    # Extract the context paragraph using regular expressions
    pattern = r'<div class="gs_rs">(.*?)<\/div>'
    paragraphs = re.findall(pattern, page_content, flags=re.DOTALL)

    citing_paragraphs = []
    for paragraph in paragraphs:
        soup = BeautifulSoup(paragraph, 'html.parser')
        plain_text = soup.get_text(separator=' ')
        plain_text = plain_text.replace("\n", "")
        citing_paragraphs.append(plain_text.strip())

    return citing_paragraphs


def find_citing_context(sentence):
    search_url = f"https://scholar.google.com/scholar?q={sentence}&hl=en&as_sdt=0,5"
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    context_list = []

    for result in soup.find_all('div', class_='gs_ri'):
        title_element = result.find('h3', class_='gs_rt')
        context_paragraph = result.find('div', class_='gs_rs')

        if title_element and context_paragraph:
            title = title_element.text.strip()
            context = get_citing_context(context_paragraph, sentence)

            if context:
                context_list.append({'title': title, 'context': context})

    return context_list


def find_citing_context(sentence):
    api_url = f"https://api.crossref.org/works?query={sentence}&rows=10"
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        articles = []

        for item in data.get('message', {}).get('items', []):
            title = item.get('title', [])
            context = get_citing_context(item, sentence)

            if context:
                articles.append({'title': title, 'context': context})

        return articles
    else:
        print("Error: Unable to retrieve citing articles.")
        return []


def get_citing_context(item, sentence):
    context = []
    doi = item.get('DOI')

    if doi:
        api_url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            full_text = data.get('message', {}).get(
                'full-text-retrieval-response', {}).get('text')

            if full_text:
                paragraphs = full_text.split('\n\n')

                for i, paragraph in enumerate(paragraphs):
                    if sentence in paragraph:
                        context.append(paragraph)
                        if i > 0:
                            context.insert(0, paragraphs[i-1])
                        if i < len(paragraphs) - 1:
                            context.append(paragraphs[i+1])
                        break

    return context


sentence = "I have a dream."

citing_context = find_citing_context(sentence)
for article in citing_context:
    print("Title:", article['title'])
    for context in article['context']:
        print("Context:", context)
    print()

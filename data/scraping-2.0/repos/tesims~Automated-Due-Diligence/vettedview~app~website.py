from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import streamlit as st
import requests
import openai
import string
import spacy
import nltk
import re


# --------- SETUP OPEN API ---------
openai.api_key = "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"

# --------- GET TEXT TOKEN SIZE ---------
def count_tokens(text):
    # Tokenize the text using the NLTK library
    tokens = nltk.word_tokenize(text)
    # Return the number of tokens
    return len(tokens)

# --------- GET ANSWERS TO QUESTIONS ---------
def summarize_key_term(key_term, text):
    summary = []
    # Tokenize the text using the NLTK library
    tokens = nltk.word_tokenize(text)
    # Divide the tokens into chunks that are fewer than 4097 tokens
    chunk_size = 3250
    num_chunks = len(tokens) // chunk_size
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    # Rejoin the tokens in each chunk into a single string of text
    text_chunks = []
    for chunk in chunks:
        text_chunk = " ".join(chunk)
        text_chunks.append(text_chunk)
    # Make an API request for each chunk of text
    for chunk in text_chunks:
        # Make an API request to the OpenAI GPT-3 model to summarize the information related to the key term

        # print(f"Summarize information related to '{key_term}': {chunk}")

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize information related to '{key_term}': {chunk}. Each key_term needs to have an answer, "
                   f"including clearly stating if any information related to a key_term was not detected. Summary "
                   f"should be detailed even if it is long.",
            max_tokens=3500,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # Process the response and extract the summary
        chunk_summary = response["choices"][0]["text"].strip().split("\n")
        summary.extend(chunk_summary)
    return summary

# Define a function to make an API request in parallel using the asyncio library
def make_request(key_term, text):
    summary = loop.run_in_executor(None, summarize_key_term, key_term, text)
    return summary

# --------- SEPARATE HTML FROM TEXT ---------
def clean_html(html):
    # parse html content
    soup = BeautifulSoup(html, "html.parser")

    for data in soup(['style', 'script', 'code', 'a']):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return ' '.join(soup.stripped_strings)

# Load spacy
nlp = spacy.load('en_core_web_sm')

# --------- REMOVE UNNECESSARY CHARACTERS ---------
def clean_string(text, stem="None"):
    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    text = re.sub(r'\n', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = nltk.corpus.stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    elif stem == 'Spacy':
        text_filtered = nlp(' '.join(text_filtered))
        text_stemmed = [y.lemma_ for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string


# Define a function to retrieve and process the main menu pages of the website
def process_menu_pages(url, key_terms):
    # Define the summarize variable
    summarize = []

    # Retrieve the main menu pages of the website
    menu_pages = requests.get(url + "/menu").text
    menu_pages = clean_html(menu_pages)

    # Search for the key terms on the main menu pages
    for key_term in key_terms:
        if key_term in menu_pages:
            # Summarize the information related to the key term
            summary = summarize_key_term(key_term, menu_pages)
            summarize.append(summary)
    return summarize


# --------- JOIN CHUNKED ANSWERS ---------

def get_url_summary(summaries, key_terms):

    result = ""
    for summary in summaries:
        for line in summary:
            result += line + "\n"
    return result


# --------- RERUN JOINED CHUNKED SUMMARY TO FIX GRAMMAR ---------
def clean_up_summary(summary,key_terms):
    # Use the OpenAI API to rewrite the summary
    prompt = f"Please revise the following summary to ensure that it has correct grammar, style, and clarity: " \
             f"{summary}. The revised summary should clearly answer all of the questions indicated by the " \
             f"'{key_terms}' about what is on the website. The response should be long and detailed."
    completions = openai.Completion.create(engine="text-davinci-003",
                                           prompt=prompt,
                                           max_tokens=3500,
                                           temperature=0.5,
                                           top_p=1,
                                           frequency_penalty=0,
                                           presence_penalty=0)

    cleaned_up_summary = completions["choices"][0]["text"]

    return cleaned_up_summary

# --------- GET FINAL ANSWERS ---------
@st.cache
def get_final_web_report(url, key_terms):
    summaries = process_menu_pages(url, key_terms)
    web_summary = get_url_summary(summaries, key_terms)
    cleaned_up_summary = clean_up_summary(web_summary, key_terms)

    return cleaned_up_summary

'''
url = 'https://soothsayer.technology/'
key_terms = ["Type",
             "Products",
             "Launched ",
             "Demo"]

temp = get_final_web_report(url, key_terms)
print(temp)
'''


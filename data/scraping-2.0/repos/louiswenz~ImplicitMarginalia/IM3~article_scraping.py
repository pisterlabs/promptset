import requests
from bs4 import BeautifulSoup
import json
import openai
import re
import nltk.data
from spellchecker import SpellChecker
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import newspaper
import time
import string
from concurrent.futures import ThreadPoolExecutor
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from google.cloud import language_v1

# open ai key (do not upload to github)
openai.api_key = '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'
openAI_key = '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'

api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"

newsapi = NewsApiClient(api_key='fa9f684c11bf41e9917bed3fe109a308')  # news api

# bing web search
subscription_key = "fb1daf6e834947edba318a368a24b620"
assert subscription_key
bing_search_url = "https://api.bing.microsoft.com/v7.0/search"


def bing_web_search(search_term, limit=10):
    search_term = "'" + search_term + "'"
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": False,
              "textFormat": "HTML", "count": limit, 'responseFilter': ['Webpages']}
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    search_results = search_results["webPages"]["value"]
    names, snippets = [], []
    for i in search_results:
        if (('act' in i["name"].lower()) and ('scene' in i["name"].lower())) or (('google translate' in i["name"].lower())):
            continue
        names.append((i["name"]))
        snippets.append(i["snippet"])
    return names, snippets


def strip_punctuation(input_string):
    # Remove leading and trailing whitespace and punctuation
    cleaned_string = input_string.strip(string.whitespace + string.punctuation)
    return cleaned_string


def check_noise_in_string(sentence, noise):
    sentence = sentence.lower()
    for n in noise:
        if n in sentence:
            return True
    return False


def beautify_string(text):
    text = text.replace('-', '')
    text = re.sub(
        r'(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z0-9])(?=[A-Z][a-z])', ' ', text)

    return text


def find_sentence_contexts(text, target_sentence):
    # Split the text into sentences
    target_sentence = target_sentence.lower()
    # text = beautify_string(text)
    sentences = nltk.tokenize.sent_tokenize(text, language='english')
    # Find the target sentence and its context
    noise = ['pdf', 'doi', 'copyright', 'https',
             'all rights reserved', 'http://', '©']
    contexts = []
    for i, sentence in enumerate(sentences):
        if check_noise_in_string(sentence, noise):
            continue
        if target_sentence in sentence.lower():
            prev_sentence = sentences[i - 1] if i > 0 else ""
            next_sentence = sentences[i + 1] if i < len(sentences) - 1 else ""
            if check_noise_in_string(prev_sentence, noise) or check_noise_in_string(next_sentence, noise):
                prev_sentence, next_sentence = '', ''
                continue
            context = prev_sentence + " " + sentence + " " + next_sentence
            context = context.strip()
            context = beautify_string(context)
            contexts.append(context)
            prev_sentence, sentence, next_sentence = '', '', ''

    return '<br><br>'.join(contexts)


def query_api(url_fragment, query, limit=2):
    headers = {"Authorization": "Bearer "+api_key}
    query = {"q": query,  "limit": limit}
    response = requests.post(
        f"{api_endpoint}{url_fragment}", data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error code {response.status_code}, {response.content}")


def core_get_results(results, target_sentence):
    articles_title = []
    articles_context = []
    for i in results['results']:
        context = find_sentence_contexts(i['fullText'], target_sentence)
        if context:
            articles_title.append(i['title'])
            context = find_most_opinionated_paragraph(context)
            context = beautify_string(context)
            articles_context.append(context)
    return articles_title, articles_context


def news_get_results(response, target_sentence):
    titles = []
    contexts = []
    for v in response['articles']:
        article = get_article_fromurl(v['url'])
        context = find_sentence_contexts(article, target_sentence)
        if context:
            titles.append(v['title'])
            context = find_most_opinionated_paragraph(context)
            context = beautify_string(context)
            contexts.append(context)
    return titles, contexts


def gpt(response):
    # Prepare the search query
    response = "<br>".join(response)
    query = f"I am looking for all the elements that are not from  King Lier by Shakespeare. I have a string where each element is separated by <br>. The string is {response}. Filter out the elements that are from the original text by Shakespeare. Return me a a string where each element is separated by <br>"

    # Issue the search request
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that summarizes text. Be descriptive. Talk in first person"},
            {"role": "user", "content": query}
        ],
        max_tokens=1000,
        stop=None,
        n=1,
        temperature=0.3
    )
    result = response["choices"][0]["message"]["content"]
    return result


def davinci03(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        n=1,
        stop='.'
    )
    return response.choices[0].text.strip()


def makeQ(target_sentence, field=''):
    # AND (_exists_:doi)
    # return f"fullText:{target_sentence} AND (_exists_:doi)"
    return f"{target_sentence} AND (_exists_:doi)"


def find_most_opinionated_paragraph(text):
    # Split text into paragraphs
    paragraphs = text.split("<br><br>")

    # Initialize Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    # Find the paragraph with the highest sentiment score
    max_sentiment_score = -1
    most_opinionated_paragraph = ""

    for paragraph in paragraphs:
        if paragraph.strip():
            sentiment_score = sia.polarity_scores(paragraph)['compound']
            if sentiment_score > max_sentiment_score:
                max_sentiment_score = sentiment_score
                most_opinionated_paragraph = paragraph

    return most_opinionated_paragraph


def find_articles(target_sentence, limit=5):
    sentence = strip_punctuation(sentence)
    core_results = query_api(
        "search/works", makeQ(target_sentence), limit=limit)
    news_response = getnewsapi(f'"{target_sentence}"', limit=limit)
    core_articles_title, core_articles_context = core_get_results(
        core_results, target_sentence)
    news_articles_title, news_articles_context = news_get_results(
        news_response, target_sentence)

    return core_articles_title+news_articles_title, core_articles_context+news_articles_context

# sentence = 'I have a dream'
# results = query_api("search/works", makeQ(sentence), limit=2)
# articles_title, articles_fulltext = get_result(results)
# contexts = [find_sentence_contexts(x, sentence) for x in articles_fulltext]
# print(contexts)

# summary = davinci03(contexts[0])
# print(summary)


def getnewsapi(q, limit=10):
    # q = '"' + q + '"'
    response = newsapi.get_everything(q=q,
                                      language='en',
                                      sort_by='popularity',
                                      page=1,
                                      page_size=limit)
    if response['status'] != 'ok':
        print("Bad Request")

    return response


def get_article_fromurl(url):
    # Initialize the Article object
    article = newspaper.Article(url)

    # Download and parse the article
    article.download()
    article.parse()

    # Return the article's main content
    return article.text

# start_time = time.time()
# sentence = 'i have a dream'
# articles_title, articles_context = find_articles(sentence)
# end_time = time.time()
# elapsed_time = end_time - start_time
# for i, v in enumerate(articles_title):
#     print(v)
#     print(articles_context[i])
#     print('----------')

# print(f"Elapsed time: {elapsed_time} seconds")


def filter_texts_by_field_of_study(texts, titles, field_of_study):
    # Prepare the prompt to instruct GPT-3 for text filtering
    prompt = f"Filter the following texts that is in the field for '{field_of_study}':\n\n"
    for text, title in zip(texts, titles):
        prompt += f"Title: {title}\n"
        prompt += f"Text: {text}\n\n"

    # Make API call to GPT-3
    response = openai.Completion.create(
        engine="text-davinci-002",  # GPT-3 engine
        prompt=prompt,
        max_tokens=1000,  # Maximum number of tokens in the response
        temperature=0.1,  # Controls the randomness of the response
    )

    # Extract the filtered texts and titles from the response
    filtered_texts_and_titles = response['choices'][0]['text'].split("\n\n")
    filtered_texts_and_titles = [x.strip() for x in filtered_texts_and_titles if x.strip()]
    if len(filtered_texts_and_titles)==0:
        print("No related contexts.")
        return ["No Reults"],["No Reults"]
    # Separate the filtered texts and titles into separate lists
    filtered_texts = []
    filtered_titles = []
    for i in filtered_texts_and_titles:
        info = i.split('\n')
        filtered_titles.append(info[0].replace("Title: ", ""))
        filtered_texts.append(info[1].replace("Text: ", ""))

    return filtered_titles,filtered_texts


example_texts = [
    # Technology
    "Artificial intelligence is revolutionizing various industries. Machine learning algorithms can now analyze big data sets to make predictions. Companies are increasingly adopting automation to streamline their processes.",

    # Sports
    "The football match was intense, with both teams giving their best performance. The striker scored a brilliant goal in the last minute. The crowd cheered loudly as the home team secured a victory.",

    # Science
    "The scientific experiment yielded unexpected results. Researchers observed a significant correlation between two variables. The findings have the potential to advance our understanding of the natural world.",

    # Travel
    "Exploring new destinations is an exciting adventure. The serene beaches and lush landscapes of Bali attract tourists from around the world. Local cuisine offers a delightful culinary experience.",

    # Food
    "The aroma of freshly baked bread filled the bakery. The chef prepared a delectable three-course meal for the guests. The restaurant's signature dessert was a hit among diners.",

    "In recent studies, scientists have discovered a new species of deep-sea creatures living in the Mariana Trench. These bioluminescent organisms emit a mesmerizing glow to attract prey in the pitch-black ocean depths. Researchers believe that the discovery could shed light on the unique adaptations of life in extreme environments and offer insights into potential biomedical applications."

]

example_titles = ['Techonolgy and science','Sports', 'Science','Travel', "Food", "Deep sea science"]


def sample_classify_text(text_content):
    """
    Classifying Content in a String

    Args:
      text_content The text content to analyze.
    """

    client = language_v1.LanguageServiceClient()

    # text_content = "That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows."

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    content_categories_version = (
        language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
    )
    response = client.classify_text(
        request={
            "document": document,
            "classification_model_options": {
                "v2_model": {"content_categories_version": content_categories_version}
            },
        }
    )
    sorted_categories = sorted(
        response.categories, key=lambda x: x.confidence, reverse=True)

    # Get the category with the highest confidence (the first category after sorting)
    highest_confidence_category = sorted_categories[0]

    # Get the name and confidence of the highest confidence category
    category_name = highest_confidence_category.name
    confidence = highest_confidence_category.confidence
    return category_name, confidence


# search_term = "Loyal and natural boy, I'll work the means"
# names, snippets = bing_web_search(search_term, limit=10)
field_of_study = "Technology"  # The specified field of study
start_time = time.time()
filtered_titles,filtered_texts = filter_texts_by_field_of_study(
    example_texts,example_titles, field_of_study)
# for i in snippets:
#     cat, prob = sample_classify_text(i)
#     print(cat)
#     print(prob)
#     print('------------')
end_time = time.time()
print(f"Elapsed time: {end_time-start_time} seconds")
print(filtered_texts)

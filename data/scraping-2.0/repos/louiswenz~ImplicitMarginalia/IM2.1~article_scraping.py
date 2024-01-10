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

# open ai key (do not upload to github)
openai.api_key = ''

api_key = "8mNcp3SVfrGqwZI9Oe0YivbxFPUhBMTk"  # core api key
api_endpoint = "https://api.core.ac.uk/v3/"

newsapi = NewsApiClient(api_key='fa9f684c11bf41e9917bed3fe109a308')  # news api


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


def summarize(contexts):
    # Prepare the search query
    summaries = []
    for i in contexts:
        query = f"Summarize the text in no more than 4 sentences: {i}"

        # Issue the search request
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes text. Be descriptive. Talk in first person"},
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            stop=None,
            n=1,
            temperature=0.3
        )
        summary = response["choices"][0]["message"]["content"]
        summaries.append(summary)
    return summaries


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


start_time = time.time()
sentence = 'i have a dream'
articles_title, articles_context = find_articles(sentence)
end_time = time.time()
elapsed_time = end_time - start_time
for i, v in enumerate(articles_title):
    print(v)
    print(articles_context[i])
    print('----------')

print(f"Elapsed time: {elapsed_time} seconds")

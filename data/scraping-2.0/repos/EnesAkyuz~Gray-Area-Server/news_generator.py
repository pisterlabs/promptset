import os

import openai
import pandas as pd
from transformers import pipeline

import news_utils
from cyber_journalist import generate_article
from news_utils import get_articles_description
from timeit import default_timer as timer

# API Key
openai.api_key = os.environ.get('OPENAI')


def keywords_from_query(user_query):
    start = timer()

    # Generate search input for news
    response = openai.Completion.create(model="text-davinci-003",
                                        prompt="Produce 3 keywords separated by semicolon for web search for this query without mentioning the source: " + user_query + ". Print them as one line. Only put quotation marks to the keywords. Do not forget to close the quotations you have opened for keywords at the end.",
                                        temperature=1, max_tokens=256)
    keywords = response.choices[0].text

    response = openai.Completion.create(model="text-davinci-003",
                                        prompt="From the list of keywords separated by semicolons: " + keywords + ". Generated from the next query: " + user_query + ". Print as one-line search query replacing semicolons with logical operators AND, OR, NOT. Put keywords in quotation marks.",
                                        temperature=0, max_tokens=256)
    query = response.choices[0].text
    end = timer()
    print(f"Keyword generation took {end - start}")
    print(query)
    return query


def sentiment_analysis(content_dict):
    start = timer()

    # Pull up a sentiment analysis out of thin air
    sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
    # apply sentiment analysis to the descriptions
    data = []
    for key in content_dict.keys():
        data.append(key)
    sentiment_dict = sentiment_pipeline(data)
    df = pd.DataFrame(sentiment_dict)
    print(df)
    end = timer()
    print(f"Sentiment analysis took: {end - start}")
    return df, data


def separate_sentiments(df):
    # seperate the positive, neutral and negative and sort them

    pos = df[df['label'] == "POS"]
    neu = df[df['label'] == "NEU"]
    neg = df[df['label'] == "NEG"]
    pos = pos.sort_values(by=['score'], ascending=False)
    neu = neu.sort_values(by=['score'], ascending=False)
    neg = neg.sort_values(by=['score'], ascending=False)
    final_pos = take10(pos)
    final_neu = take10(neu)
    final_neg = take10(neg)
    pos_indexes = list(final_pos.index.values)
    neu_indexes = list(final_neu.index.values)
    neg_indexes = list(final_neg.index.values)

    return pos_indexes, neu_indexes, neg_indexes


# take the first 10 of each list and grab the corresponding content for them
def take10(df):
    return df.head(10)


def content_grabber(index_list, data):
    grabbed = ""
    for elem in index_list:
        grabbed += data[elem]
    return grabbed


def generate_news(query):
    content_dict = get_articles_description(keywords_from_query(query))
    print(content_dict)
    df, data = sentiment_analysis(content_dict)

    if not df.empty:
        pos_indexes, neu_indexes, neg_indexes = separate_sentiments(df)

        pos_grabbed = content_grabber(pos_indexes, data)
        neu_grabbed = content_grabber(neu_indexes, data)
        neg_grabbed = content_grabber(neg_indexes, data)

        sentiments = {
            "pro": pos_grabbed,
            "neu": neu_grabbed,
            "neg": neg_grabbed
        }

        print(sentiments)
        return sentiments
    return None


def news_generation(query):
    start = timer()

    source = generate_news(query)
    if source is None:
        print('Failed search for: ' + query)
        return
    article = generate_article(source, True)

    end = timer()
    print(f"News generation took:{end - start}")


def top_newsletter():
    start = timer()

    top = news_utils.get_top_titles()

    for q in top:
        news_generation(q)
    end = timer()
    print(f"News letter generation took:{end - start}")

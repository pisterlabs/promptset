"""
This Python module contains all the functions and classes for obtaining the sentiment for an article. This provides
the next step by transforming the scraped and raw data into numbers - usable sentiment scores
"""
from __future__ import annotations
from typing import Optional
import openai
from openai.error import RateLimitError
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, BertForSequenceClassification, BertTokenizer
from nltk.corpus import stopwords
from python_ta.contracts import check_contracts
from StockInfo import Stock
from NewsScraper import NewsArticleContent
from dataclasses import dataclass, field
import ast
import time
import random
import StockInfo
import nltk

# intialize vader sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# intialize finbert
finbert_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_get_sentiment = pipeline("text-classification", model=finbert_model, tokenizer=finbert_tokenizer)
MAX_FINBERT_TOKENS = 512
FINBERT_LABELS = {
    'Positive': 7,
    'Neutral': 0,
    "Negative": -7
}

# model constants
openai.api_key = "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"
model_engine = "gpt-3.5-turbo"
PROMPT_ERROR = 'ERROR'
SET_UP_PROMPT = "Give a sentiment score from -10 to 10 for each company that is public on the market " \
                "in a " \
                "dictionary format with the companys' ticker as the keys. Do NOT provide any other output. Output " \
                + PROMPT_ERROR + " on any errors.\n"
MAX_TOKENS = 250
OPENAI_MAX_REQUESTS = 3


@dataclass
class ArticleSentimentData:
    """A dataclass represetning the data returned by sentiment

    Instance Attributes:
        - main_sentiment_score: a float -10 <= x <= 10. This score represents the main stock's sentiment
        - other_sentiment_scores: a dictionary representing other stocks mentioned in the article corresponding to
                                  their sentiment.
    """

    main_sentiment_score: float
    other_sentiment_scores: dict[str, float]


def get_complex_phrase_sentiment_score(passage: str) -> dict[str, float]:
    """
    Used when retrieving the sentiment scores of multiple companies in a singular sentence/paragraph.
    """
    time.sleep(random.uniform(0.05, 0.2))
    request_tries = 1
    result = None
    while request_tries <= OPENAI_MAX_REQUESTS and result is None:
        try:
            response = openai.ChatCompletion.create(
                model=model_engine,
                messages=[{"role": "system", "content": SET_UP_PROMPT + passage}],
                max_tokens=MAX_TOKENS,
                temperature=0,
                stop=None,
            )
            result = response.choices[0].message.content
        except RateLimitError:
            time.sleep(0.5)
            print("Retry CHAT-GPT Api Call")
            request_tries += 1
    if result is None:
        # somethign went wrong so return an empty dictioanry
        return {}
    if result == PROMPT_ERROR:
        # somethign went wrong so return an empty dictionary
        return {}
    else:
        # return the result but parsed as a dictionary
        if result[0] != '{' or result[len(result) - 1] != '}':
            # edge case of chat-gpt returning incorrect info.
            return {}
        try:
            response = ast.literal_eval(result)
            tickers = StockInfo.get_tickers()
            temp = list(response.keys())
            for key in temp:
                if key not in tickers:
                    response.pop(key)
            return response
        except (SyntaxError, ValueError):
            # some decoding went wrong so return an empty dictionary
            return {}


def get_sentiment_single(passage: str) -> float:
    """
    Returns the sentiment score for a passage ASSUMING THERE IS ONLY ONE STOCK MENTIONED IN THE PASSAGE
    Uses nltk's VADER and FinBert

    Preconditions:
        - there is only ONE stock in the passage
    """
    # clean text by filtering out words that typically do not carry much meaning such as "and","the", "of"
    # removing this "fluff" may improve accuracy, but also may not, hence this function will average it
    stop_words = stopwords.words("english")
    cleaned_text = ' '.join([word for word in passage.split() if word not in stop_words])
    # return sentiment of the passage which is the average compound scores for the raw passage and the cleaned one
    cleaned_score = sentiment_analyzer.polarity_scores(cleaned_text)['compound']
    raw_score = sentiment_analyzer.polarity_scores(passage)['compound']
    vader_score, finbert_score = (cleaned_score + raw_score) * 5, 0
    if len(finbert_tokenizer.tokenize(passage)) < MAX_FINBERT_TOKENS:
        # make sure the sentence isn't too long for finbert
        finbert_score = FINBERT_LABELS[finbert_get_sentiment(passage)[0]['label']]
    # calculate overall sentiment score while siding more with finbert's score as its more accurate
    print(vader_score)
    print(finbert_score)
    sentiment_score = finbert_score * 0.65 + vader_score * 0.35
    return sentiment_score


def get_stocks_in_passage(passage: str) -> set:
    """
    Returns a set containing all the stocks mentioned in the passage as a ticker
    """
    stocks_mentioned = set()
    tickers, names = StockInfo.get_tickers_and_names()
    passage = ' ' + passage

    # for names, use all same casing (upper case)
    sentence = passage.upper()
    for name in names:
        if " " + name + " " in sentence or " " + name + "." in sentence:
            stocks_mentioned.add(StockInfo.get_ticker_from_name(name))

    return stocks_mentioned


def get_sentiment_for_article(main_stock: Stock, news_article: NewsArticle) -> ArticleSentimentData:
    """
    Returns the sentiment data for an article.
    Assumes main_stock is the stock that is mainly being analyzed here

    Preconditions:
        - news_article is a NewsArticle object that has finished the newscraping process
    """
    title_stocks = get_stocks_in_passage(news_article.title)
    title_stock_score = 0
    sentiment_data = {}
    if len(title_stocks) > 1:
        sentiment_data.update(get_complex_phrase_sentiment_score(news_article.title))
    elif len(title_stocks) == 1:
        sentiment_data[title_stocks.pop()] = get_sentiment_single(news_article.title)
    if main_stock.ticker in sentiment_data:
        title_stock_score = sentiment_data.pop(main_stock.ticker)  # don't want main stock to be in other stocks dict
    content = news_article.sentences
    passage_stock_score = 0
    sentence_counter = 0
    for passage in content:
        passage_stocks = get_stocks_in_passage(passage)

        if len(passage_stocks) > 1:
            passage_stocks_sentiment = get_complex_phrase_sentiment_score(passage)
            for stock in passage_stocks_sentiment:
                if stock in sentiment_data:
                    sentiment_data[stock] = (sentiment_data[stock] + passage_stocks_sentiment[stock]) / 2
                else:
                    sentiment_data[stock] = passage_stocks_sentiment[stock]
            sentence_counter += 1
        elif len(passage_stocks) == 1:
            sentiment = get_sentiment_single(passage)
            if sentiment != 0:
                # ignore sentiment values that are baseline neutral as it dilutes the average too much
                sentiment_data[passage_stocks.pop()] = sentiment
                sentence_counter += 1
        if main_stock.ticker in sentiment_data:
            passage_stock_score += sentiment_data.pop(main_stock.ticker)

    # adjustment for main stock - title is more heavily weighted
    if sentence_counter == 0:
        main_stock_score = title_stock_score
    else:
        main_stock_score = (title_stock_score * 0.60) + (passage_stock_score * 0.40) / sentence_counter
    # adjustment for other stocks since the article is not primarly focused on the other stocks, make it weigh
    # slightly less
    for stock in sentiment_data:
        sentiment_data[stock] *= 0.8
    return ArticleSentimentData(main_sentiment_score=main_stock_score, other_sentiment_scores=sentiment_data)

if __name__ == '__main__':
    import doctest
    import python_ta

    doctest.testmod(verbose=True)

    python_ta.check_all(config={
        'max-line-length': 120,
        'extra-imports': ['__future__', 'typing', 'openai', 'openai.error', 'nltk.sentiment', 'transformers', 'nltk.corpus',
                          'NewsScraper', 'dataclasses', 'ast', 'time', 'random', 'StockInfo', 'nltk'],
        'allowed-io': ['NewsScraper.scrape_articles'],
        'max-nested-blocks': 10
    })

import openai
import statistics
import web_scraping_toolkit as ws
import time
from config import get_config_params

params = get_config_params()

if params == 'OpenAI API Key not set':
    raise Exception('OpenAI API Key not set, please set that in config.py')
else:
    GPT_API_KEY = params[0]
    free_version = params[1]
    

def generate_stock_sentiment_scores(ticker, start = None):
    counter = 0
    if start != None:
        news = ws.scrape_stock_news(ticker, date= start)
    else:
        news = ws.scrape_stock_news(ticker)
    scores = {}
    for date in news.keys():
        if len(news[date]) == 2:
            headline, summary = news[date]
            new_score = gpt_sentiment_analysis([ticker, headline, summary], MODE= 'STOCK_NEWS')

            if new_score != 0:
                scores[date] = new_score
            counter += 1
            if free_version and counter % 3 == 0:
                time.sleep(60)
    if free_version:
        time.sleep(60)
    return scores

def generate_market_sentiment_scores(link = None):
    if link != None:
        summary, date, paragraphs = ws.scrape_market_news(link= link, transcript= True)
    else:
        summary, date, paragraphs = ws.scrape_market_news()
    scores = []
    for para in paragraphs:
        scores.append(gpt_sentiment_analysis([summary, para], MODE= 'MARKET'))
    return scores, date

def generate_stock_supplement_scores(tickers):
    score_dict = {}
    counter = 0
    blurb_dict = ws.scrape_stock_schwab(tickers)
    for date in blurb_dict.keys():
        date_blurbs = blurb_dict[date]
        date_scores = {}
        for ticker in date_blurbs.keys():
            ticker_scores = []
            ticker_blurbs = date_blurbs[ticker]
            for blurb in ticker_blurbs:
                score = gpt_sentiment_analysis([ticker, blurb], MODE = 'STOCK_SUPP')
                ticker_scores.append(score)
                counter += 1
                if free_version and counter % 3 == 0:
                    time.sleep(60)
            date_scores[ticker] = statistics.mean(ticker_scores) if len(ticker_scores) > 0 else 0
        score_dict[date] = date_scores
    return score_dict

def gpt_sentiment_analysis(content, MODE):
    openai.api_key = GPT_API_KEY
    print(content)
    if MODE == 'MARKET':
        summary = content[0]
        paragraph = content[1]
        messages = [
        {'role': 'user', 'content': 'You will work as a Sentiment Analysis for Financial News. I will share a summary and a paragraph from an article about the current state of the market. You will respond with a score of how bullish or bearish the content of the paragraph is using the context of the summary. Your response should be only the score and no other text. The range is -10 for the most bearish to 10 for the most bullish. Got it?'},
        {'role': 'system', 'content': 'Got it! Please go ahead and provide me with the summary and the paragraph from the article, and I will provide you with a score indicating the bullishness or bearishness of the content.'},
        {'role': 'user', 'content': 'Summary: ' + summary + '\n' + 'Paragraph: ' + paragraph}]
    elif MODE == 'STOCK_NEWS':
        ticker = content[0]
        headline = content[1]
        summary = content[2]
        messages = [
        {'role': 'user', 'content': 'You will work as a Sentiment Analysis for Financial News. I will share a stock name, a news headline, and a summary, and you will respond with a score of how bullish or bearish it is. Your response should be only the score and no other text. The range is -10 for the most bearish to 10 for the most bullish. If the headline and summary are unrelated to the stock, you will report a neutral score of 0. No further explanation. Got it?'},
        {'role': 'system', 'content': 'Got it! Please provide me with the stock name, news headline, and summary, and I will provide you with a sentiment score ranging from -10 (most bearish) to 10 (most bullish).'},
        {'role': 'user', 'content': 'Stock: ' + ticker + '\n' + 'Headline: ' + headline + '\n' + 'Summary: ' + summary}]
    elif MODE == 'STOCK_SUPP':
        ticker = content[0]
        blurb = content[1]
        messages = [
        {'role': 'user', 'content': 'You will work as a Sentiment Analysis for Financial News. I will share a stock name and a blurb which contains some information relevant to the stock. You will respond with a score of how bullish or bearish the information relevant to the stock is. Your response should be only the score and no other text. The range is -10 for the most bearish to 10 for the most bullish. If the headline and summary are unrelated to the stock, you will report a neutral score of 0. No further explanation. Got it?'},
        {'role': 'system', 'content': 'Got it! Please provide me with the stock name and the blurb, and I will provide you with a sentiment score ranging from -10 (most bearish) to 10 (most bullish).'},
        {'role': 'user', 'content': 'Stock: ' + ticker + '\n' + 'Blurb: ' + blurb }]

    response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages= messages, temperature= 0,)
    
    score = response['choices'][0]['message']['content']
    if score[0] == 'S':
        score = score[-1]
    return float(score)
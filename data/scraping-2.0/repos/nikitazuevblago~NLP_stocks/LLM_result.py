import numpy as np
from statistics import mean
import spacy
from datetime import date
from spacytextblob.spacytextblob import SpacyTextBlob
import warnings
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from secret_key import api_key, api_key_finance
warnings.filterwarnings('ignore')



def NLP_stocks(tick):

    def get_stock_news(date, stock, api_key) -> pd.DataFrame:
        date_7_ago = date - timedelta(days=7)
        date_7_ago = date_7_ago.strftime('%Y-%m-%d')
        url = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={stock}&from={date_7_ago}&to={date}&limit=1000'
        news_json = requests.get(url).json()
        news_data = pd.DataFrame(news_json)
        news_data = news_data.drop(['sentiment'], axis = 1)
        news_data['date'] = news_data['date'].apply(lambda x: datetime.fromisoformat(x).strftime(r'%Y-%m-%d'))
        d = {date:[] for date in news_data['date'].unique()}
        for index,row in news_data.iterrows():
            row_date = row['date']
            d[row_date].append(row['title'])
        date_col = [key for key, value in d.items()]
        news_col = [value for key, value in d.items()]
        return pd.DataFrame({'date':date_col,'news_col':news_col})

    today = date.today()

    news_data = get_stock_news(today, tick, api_key_finance)


    def adding_sent_subj(df, ticker):
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe('spacytextblob')

        daily_sentiment_col = []
        daily_subjectivity_col = []
        for index, row in df.iterrows():
            daily_sentiment_list = []
            daily_subjectivity_list = []
            for news in row['news_col']:
                doc = nlp(news)
                daily_sentiment_list.append(doc._.blob.polarity)
                daily_subjectivity_list.append(doc._.blob.subjectivity)
            daily_sentiment_col.append(np.mean(daily_sentiment_list))
            daily_subjectivity_col.append(np.mean(daily_subjectivity_list))
        df['daily_sentiment'] = daily_sentiment_col
        df['daily_subjectivity'] = daily_subjectivity_col

        df.to_csv(f'data/{ticker}.csv')
        print('Saving dataset to "data" folder!\n\n')

        return df[::-1]


    news_data_sent_subj = adding_sent_subj(news_data, tick)

    llm = ChatOpenAI(openai_api_key=api_key,
                                model_name='gpt-3.5-turbo-16k',    # gpt-3.5-turbo | gpt-3.5-turbo-16k  | gpt-4
                                temperature=0.3)


    def subj_sent_analyzer(df):
        template = ChatPromptTemplate.from_messages(
            [
                # SystemMessage(content=("Тебе нужно дать ответ ввиде python списка. Пример: ['item1', 'item2', 'item3'...]")),
                HumanMessagePromptTemplate.from_template(
                    "Imagine you're a financial assistant. You've been given news for recent 2 days. Also you've been given sentiment and subjectivity rating of these news. Your task is to analyze future stock trend by these data.\n"
                    "Sentiment for day 1: {SENTIMENT_1}\n"
                    "Subjectivity for day 1: {SUBJECTIVITY_1}\n"
                    "Sentiment for day 2: {SENTIMENT_2}\n"
                    "Subjectivity for day 2: {SUBJECTIVITY_2}\n")])

        response = llm(template.format_messages(
            SENTIMENT_1=df['daily_sentiment'].iloc[0],
            SUBJECTIVITY_1=df['daily_subjectivity'].iloc[0],
            SENTIMENT_2=df['daily_sentiment'].iloc[1],
            SUBJECTIVITY_2=df['daily_subjectivity'].iloc[1])).content
        return response


    subj_sent_response = subj_sent_analyzer(news_data_sent_subj)


    def news_analyzer(df):
        template = ChatPromptTemplate.from_messages(
            [
                # SystemMessage(content=("Тебе нужно дать ответ ввиде python списка. Пример: ['item1', 'item2', 'item3'...]")),
                HumanMessagePromptTemplate.from_template(
                    "Imagine you're a financial assistant. You've been given lists of news for recent 2 days. Your task is to analyze future stock trend by these data.\n"
                    "List of news of the first day: {NEWS_1}\n"
                    "List of news of the second day: {NEWS_2}\n")])

        response = llm(template.format_messages(
            NEWS_1=df['news_col'].iloc[0],
            NEWS_2=df['news_col'].iloc[1])).content
        return response


    news_response = news_analyzer(news_data_sent_subj)

    return subj_sent_response+'\n\n'+news_response

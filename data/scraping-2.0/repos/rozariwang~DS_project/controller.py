from datetime import datetime, timedelta

import openai
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import yahoo_fin.stock_info as si
from stocksent.sentiment import Sentiment
from transformers import AutoModelForSequenceClassification  # type: ignore
from transformers import AutoTokenizer  # type: ignore

# Set up OpenAI API credentials
openai.organization = st.secrets["openai_organization"]
openai.api_key = st.secrets["openai_api_key"]


@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    """
    Loads the FinBERT model for sentiment analysis.
    """
    return AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

@st.cache_resource  # 👈 Add the caching decorator
def load_tokenizer():
    """
    Loads the FinBERT tokenizer for sentiment analysis.
    """
    return AutoTokenizer.from_pretrained("ProsusAI/finbert")

@st.cache_data
def load_companies():
    """
    Loads the company dictionary from the csv file.
    Returns:
        dict: A dictionary of companies.
    """
    companies = pd.read_csv('ticker_lookup_df.csv', names=['ticker', 'name'], header=0)
    return {k: v for k, v in zip(companies["ticker"], companies["name"])}

# define the Network

class MyNetwork(nn.Module):
    def __init__(self, lr=0.0001):
        super(MyNetwork, self).__init__()
        self.learning_rate = lr

        self.network = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
        nn.Sigmoid()
)

    def forward(self, x):
        return self.network(x)

def get_sentiment(input_text: str) -> "list[float]":
    """
    Performs sentiment analysis on the input text using the FinBERT model.
    Args:
        input_text (str): The text for sentiment analysis.
    Returns:
        list: A list of sentiment probabilities for each class.
    """
    tokenizer = load_tokenizer()
    model = load_model()
    inputs = tokenizer(input_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    return torch.nn.Softmax(dim=1)(logits)[0].tolist()

def get_stock_data(ticker: str) -> "tuple[torch.tensor(), float, str]|int": # type: ignore
    """
    Retrieves stock data, performs sentiment analysis on related news stories,
    and returns the processed data as a tuple.
    Args:
        ticker (str): The stock ticker symbol.
    Returns:
        tuple: A tuple containing the processed stock data tensor and the annual percent change.
    """
    today: datetime = datetime.today()
    yesteryear: str = (today - timedelta(days=345)).strftime('%Y-%m-%d')
    column_names: "list[str]" = ["low","open","volume","high","close","adjclose","Annual Percent Change","positive","negative","neutral"]
    stock_means: "list[float]" = [526.3256709784156, 543.3051050285828, 1550317.649111653, 560.6335547940919, 543.4596727527701, 432.9478474861315, 2.1972159402834595, 0.26918878308244654, 0.27109889272321963, 0.45971232487752245]
    stock_std: "list[float]" = [25100.799536177463, 25956.76629687626, 13374912.758878224, 26792.649346820093, 25956.514595704837, 20572.229772641494, 477.38693948841035, 0.13956166294058042, 0.1719876441524078, 0.16237148015431316]

    mean_series = pd.Series(stock_means, index=column_names)
    std_series = pd.Series(stock_std, index=column_names)

    try:
        stock_news = Sentiment(ticker)
        sentiment_score = stock_news.get_dataframe(days=1)
        stories = sentiment_score['headline'].tolist()[:10]
        sentiments: "list[list[float]]" = []

        for story in stories:
            sentiment = get_sentiment(story)
            sentiments.append(sentiment)
        sentiments_df = pd.DataFrame(sentiments, columns=['positive', 'negative', 'neutral'])
        mean_sentiments: pd.Series = sentiments_df.mean()
        sentiment_return: str = str(pd.Series.idxmax(mean_sentiments))
        stock_sentiment: "list[float]" = mean_sentiments.values.tolist()
    except:
        sentiment_return = "neutral"
        stock_sentiment = [0.3, 0.3, 0.4]
    try:
        ticker_info: pd.DataFrame = si.get_data(ticker, start_date=yesteryear, end_date=today, interval="1mo")
    except:
        return 404
    ticker_info = ticker_info.drop(columns=['ticker'])
    ticker_info = ticker_info.fillna(ticker_info.mean())
    ticker_info['Annual Percent Change'] = (ticker_info.iloc[-1]['close'] - ticker_info.iloc[0]['close']) / ticker_info.iloc[0]['close'] * 100
    annualized_ticker_info = ticker_info.mean()

    annualized_ticker_info['positive'] = stock_sentiment[0]
    annualized_ticker_info['negative'] = stock_sentiment[1]
    annualized_ticker_info['neutral'] = stock_sentiment[2]
    annualized_ticker_info = annualized_ticker_info[column_names] # type: ignore

    normalalized_ticker_info = (annualized_ticker_info - mean_series) / std_series

    return torch.Tensor(normalalized_ticker_info.values.tolist()), annualized_ticker_info['Annual Percent Change'], sentiment_return


def generate_stock_prediction(company: str) -> "tuple[float, float, str]|int":
    """
    Generates a stock prediction for a given company using the retrieved stock data and a loaded model.
    Args:
        company (str): The name or symbol of the company.
    Returns:
        tuple: A tuple containing the stock prediction and the annual percent change.
    """
    try:
        stock_data, annual_percent_change, sentiment = get_stock_data(company)
    except:
        return 404
    model = MyNetwork()
    model.load_state_dict(torch.load("ProfifPropheNet-v1.pt"))
    model.eval()
    with torch.no_grad():
        prediction = model(stock_data).round()
    return prediction.item(), annual_percent_change, sentiment

# Function to generate recommendation using ChatGPT API
@st.cache_data
def generate_recommendation(company: str):
    """
    Generates a recommendation for a given company based on the stock prediction and annual percent change.
    Args:
        company (str): The name or symbol of the company.
    Returns:
        str: The generated recommendation as a response to the prompt.
    """
    company_dict = load_companies()
    company_ticker = company_dict[company]

    try:
        prediction, annual_percent_change, sentiment = generate_stock_prediction(company_ticker)
    except:
        return 404
    # if annual_percent_change == "nan%":
    #     return 404
    # if type(annual_percent_change) != float:
    #     return 404

    
    prompt = str(f"""Given the score 0=do not invest, and 1=invest, our classifier model gives company {company_ticker} a score of {round(prediction, 2)}.
                 The model is based on historical stock data and news headline sentiment (historical and current). The decision parameter is based on the long-term performance of this stock (average annual stock value percentage change), and whether the characteristics of this company align with those that tend to perform above the inflation threshold. 
                 Based on this score, provide a short recommendation of whether or not the user should invest in this company as a long-term investment. Include the following company metrics in the response: average annual percentage change of {round(annual_percent_change, 2)}% and current {sentiment} sentiment of news articles for this company.
                 The explanation should be understood by someone new to investing. Start the response with the explanation and then give the decision parameter. It should be clear that the numeric stock data & sentiment data are both factors which contribute to the final classification.
                 Limit the response to 200 words."""
                )
    
    completion = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.5
    )
    response = completion.choices[0].text # type: ignore
    return response

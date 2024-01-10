import json
import re
import pandas as pd

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from datetime import datetime
from typing import Dict, List
from tweety.types import Tweet

PROMPT_TEMPLATE = """
You are an AI Expert with 10+ years of experience. You always follow the trend
and follow and deeply understand AI experts on Twitter. You always consider the historical statements for each expert on Twitter.

You're given tweets from @{twitter_handle} for specific dates:

{tweets}

Tell how they feel about the risks and safety of AI. Use numbers between 0 and 100, 
where 0 indicates a lack of AI risk and 100 indicates a high level of safety concern 
and the need for regulation.

Use a JSON using the format:

date: sentiment

Each record of the JSON should give the aggregate sentiment for that date. Return just the JSON. Do not explain.
"""

def clean_tweet(tweet:str)->str:
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"www.\S+", "", tweet)
    tweet = re.sub(r"\s+", " ", tweet)
    return tweet

def create_df_from_tweets(tweets: List[Tweet]) -> pd.DataFrame:
    rows = []
    for t in tweets:
        clean_text = clean_tweet(t.text)
        if(len(clean_text) == 0):
            continue

        rows.append(
            {
                "id": t.id,
                "text": clean_text,
                "author": t.author.username,
                "date": str(t.date.date()),
                "created_at": t.date,
                "views": t.views,
                "followers": t.author.followers_count
            }
        )

    df = pd.DataFrame(
        rows, 
        columns=[
            "id", 
            "text", 
            "author", 
            "views", 
            "followers",
            "date", 
            "created_at"
        ]
    )
    df.set_index("id", inplace=True)
    if df.empty:
        return df
    df = df[df.created_at.dt.date > datetime.now().date() - pd.to_timedelta("7day")]
    df = df.sort_values(by="created_at", ascending=False)
    return df

    
def create_tweet_list_for_prompt(tweets: List[Tweet], twitter_handle: str) -> str:
    df = create_df_from_tweets(tweets)
    user_tweets = df[df.author == twitter_handle]
    if user_tweets.empty:
        return ""
    if len(user_tweets) > 100:
        user_tweets = user_tweets.sample(n=100)

    text = ""

    for tweets_date, tweet in user_tweets.groupby("date"):
        text += f"{tweets_date}:"
        for tw in tweet.itertuples():
            text += f"\n{tw.text}"
    
    return text

def analyze_sentiment(twitter_handle: str, tweets: List[Tweet]) -> Dict[str, int]:
    chat_gpt = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    prompt = PromptTemplate(
        input_variables=["twitter_handle", "tweets"], template=PROMPT_TEMPLATE
    )

    sentiment_chain = LLMChain(llm=chat_gpt, prompt=prompt)
    response = sentiment_chain(
        {
            "twitter_handle": twitter_handle,
            "tweets": create_tweet_list_for_prompt(tweets, twitter_handle),
        }
    )
    return json.loads(response["text"])
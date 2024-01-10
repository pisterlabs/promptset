import openai 
import chromadb   
import os
import re
import time

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain 

from collections.abc import Iterable, Container
from pprint import pprint

# use python-dotenv to get API key
from dotenv import load_dotenv
load_dotenv()



# Helper functions
def replace_ticker(queries, ticker: str) -> list:
    return [query.replace('{ticker}', ticker) for query in queries]


# User queries - batch single queries into a list; split across 2 assets - ETH & BTC
assets = ["ETH", "BTC"]

# Q1: Price Changes
# Q2-3: Volume
# Q4-6: RSI
# Q7-9: Market Cap
# Q10: Market Dominance
# Q11-13: News
user_queries = [
    "What are the biggest price changes of {ticker} in the last 30 days?", 
    "What is the level of {ticker} buying over the past 30 days?",
    "What is the level of {ticker} selling over the past 30 days",
    "How does the level of {ticker} buying compare to the level of {ticker} selling for the past month?",
    "How quickly has {ticker} price moved over the past 30 days?",
    "How volatile is {ticker} trading in the past month?",
    "What is the current total dollar market value of {ticker}?",
    "Has {ticker} reached its all-time high recently?",
    "How close is {ticker} to its ATL?",
    "How does the current total dollar market value of {ticker} compare to other projects?",
    "Are there any upcoming major vesting events for {ticker}?",
    "Are there any updates from the core developers for {ticker}?"
]

# user helper function to create two lists of queries for each ETH & BTC
all_queries_assets = [ replace_ticker(user_queries, asset) for asset in assets ]

#pprint(all_queries_assets)

# Take two list of queries for ETH & BTC
# run the openai.Completion.create() function over each of them
# need to insert an initial prompt to OpenAI
# This prompt Expands the initial one-line questions into a more comprehensive interpretation of user intent

def prompt(query: str) -> str:
    return f"""
    Role: You are a language model metric dispatcher for a crypto analyst. Your task is to interpret and expand user queries related to cryptocurrency quantitative metrics. 
    The expanded form should provide a more comprehensive interpretation of the user's intent and the type of data that would best address their question.
    Instruction: Taker the user's query, interpret its underlying intention, and expand it to include more detail and context. Your expansion should indicate what kind of data, metrics, or insights would best addres the user's query. 
    Remember to keep the language clear, descriptive, and related to the user's original question. Provide the ticker, description of the metric, your current user case, and how it gets used.
    Type of metrics you have access to are: Techical Analysis indicators, price metrics, token distribution, company token vesting, official social announcement channels, thread discussion on twitter.

    Example:

    Input:
    User query:
    "What is the current price of $ETH?"

    Output:
    Metric("$ETH", "The user is interested in the current market price of the $ETH token. They're likely trying to get a quick understanding of its present valuation. The metric should simply present the most up-to-date market price for $ETH today")

    Input:
    User query:
    "{query}"

    Output:
    """

openai.api_key = os.environ.get("OPENAI_API_KEY")

def open_ai_prompt(prompt: str, temp=0.3) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2047,
        temperature=temp
    )
    return response.choices[0].text.strip()


user_and_llm_queries = []

for queries in all_queries_assets:
    expanded_queries = [ open_ai_prompt(prompt(q)) for q in queries]
    #print(expanded_queries)
    for index, eq in enumerate(expanded_queries):
        user_and_llm_queries.append({"query": queries[index], "expansion": eq})

pprint(user_and_llm_queries)

# Revised Questions to reflect MVP Engineering Spec
# Price Changes, RSI, ETH, Volume, Marketcap, Market Dominance, News
# create helper functions to turn Openai-Expanded questions (intention)
# into individual tools


# Price Change includes magnitude of price changes. 
# note: combining 3 metrics into 1 implies 3 user queries should be matched to this one function
# Volume includes: Buy and Sell volume
# RSI include speed and volatiltiy of price changes. 

def price_change(ticker: str) -> str:
    return f"""Name: {ticker} Price Change ({ticker}PC)
    Description: The ticker Price Change, or {ticker}PC, measures the biggest price changes of the ${ticker} token over the past 30 days including highest and lowest values. 
    This metric should provide insight into range of price movements for ${ticker} over the past 30 days; this could be percentage change between highest and lowest values.
    When to use: Use {ticker}PC to get a better understanding of the biggest price changes of {ticker} in the past 30 days.
    """

def volume(ticker: str) -> str:
    return f"""Name: {ticker} Exchange Trade Volume Metrics ({ticker}VOL)
    Description: This metric provides a comprehensive overview of buying and selling activity of ${ticker} over the past 30 days, across major exchanges.
    This metric includes total number of buyers, the total amount of ${ticker} purchased and sold, compared to total supply. 
    Finally, volume should be calculated at the average price of ${ticker} over the past 30 days.
    When to use: Use {ticker}VOL to get get a better understanding of current sentiment of the ${ticker} market, context for current price of ${ticker}, 
    and potential for future price movements. 
    """


def rsi(ticker: str) -> str:
    return f""" {ticker} Relative Strength Index ({ticker}RSI)
    Description: The {ticker} Relative Strength Index, or {ticker}RSI, shows the relative levels of buying and selling volume for {ticker} over the past month.
    This metric should provide a comparison of the total {ticker} bought and sold over the past 30 days, ideally with a visual representation of the data.
    This indicator shows how fast prices of {ticker} has changed over the past 30 days. Traditionally, the ${ticker} token is considered overbought when the {ticker}RSI is above 70 and oversold when it's below 30.
    When to use: Use {ticker}RSI when you want to identify potentially overbought or oversold conditions for the ${ticker} token.
    {ticker}RSI may indicate how quickly {ticker} price has moved. This can help inform decisions about when to buy or sell the token.
    """

def market_cap(ticker: str) -> str:
    return f""" {ticker} Market Cap
    Description: The {ticker} Market Cap, or {ticker}MC, shows the total dollar market value of ${ticker}. 
    The metric should provide the total dollar market value of ${ticker}, as well as a comparison to its all-time high price. 
    Additionally, the total dollar market value of ${ticker} should also be compared to its all-time-low price.
    When to use: Use {ticker}MC to understand the total market capitalization of the token and whether or not it has recently reached its
    all-time highs or all-time lows.
    """

def market_dom(ticker: str) -> str:
    return f""" {ticker} Market Dominance ({ticker}MD)
    Description: This metric shows the current total dollar market value of {ticker} compared to other projects. 
    When to use: Use {ticker}MD to gain a better understanding of {ticker}'s relative value compared to other projects. 
    """

def news(ticker: str) -> str:
    return f"""Notable News for {ticker}
    Description: The {ticker}NEWS metric gathers recent tweets and Twitter threads for notable events related to the {ticker} token.
    This includes notable information about major vesting events, project updates, announcements, and discussions surrounding the token from core developers and official team sources. 
    This can include everything from exchange listings, collaborations, partnerships, to major events and milestones and product updates.
    When to use: This information can be used to inform investors of potential market movements related to the release of {ticker} tokens.
    """



# ChromaDB add collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="feeds")

for i, ticker in enumerate(assets):
    collection.add(
        documents=[price_change(ticker), volume(ticker), rsi(ticker), market_cap(ticker), market_dom(ticker), news(ticker)],
        metadatas=[{"asset": ticker}, {"asset": ticker}, {"asset": ticker}, {"asset": ticker}, {"asset": ticker}, {"asset": ticker}],
        ids=[ ticker+str(ind) for ind in range(1,7)]
    )

outputs = []

for q in user_and_llm_queries:
    results = collection.query(
        query_texts=[q["query"], q["expansion"]],
        n_results=1,
    )
    print('\n')
    print('user query: ', q["query"])
    print('expansion query: ', q["expansion"])
    print('ids:', results["ids"])
    print('documents:', results["documents"])
    print('distances:', results["distances"])
    outputs.append({"user_query": q["query"],
                    "expansion_query": q["expansion"],
                    "ids": results["ids"],
                    "documents": results["documents"],
                    "distances": results["distances"]
                    })



# langchain to connect query to SQLite
db = SQLDatabase.from_uri("sqlite:///../generate_data/demo.db")

llm = OpenAI(temperature=0, verbose=True)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# What are the biggest price changes of ETH in the last 30 days?
#db_chain.run(outputs[0]['user_query'])

#db_chain.run([outputs[i]['user_query'] for i in range(len(outputs))])

# slow down
for i in range(len(outputs)):
    result = db_chain.run([outputs[i]])
    print(result)
    print('\n')
    time.sleep(1)
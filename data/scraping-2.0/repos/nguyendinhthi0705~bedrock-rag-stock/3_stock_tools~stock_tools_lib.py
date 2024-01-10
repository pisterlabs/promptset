import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import json
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
from datetime import date
from langchain.prompts.prompt import PromptTemplate
import yfinance as yf
yf.pdr_override() 

def get_llm():
        
    model_parameter = {"temperature": 0.0, "top_p": .5, "max_tokens_to_sample": 2000}
    llm = Bedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"), #sets the profile name to use for AWS credentials (if not the default)
        region_name=os.environ.get("BWB_REGION_NAME"), #sets the region name (if not the default)
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"), #sets the endpoint URL (if necessary)
        model_id="anthropic.claude-v2", #set the foundation model
        model_kwargs=model_parameter,
        streaming=True)

    return llm

def get_db_chain(prompt):
    db = SQLDatabase.from_uri("sqlite:///stock_ticker_database.db")
    llm = get_llm()
    db_chain = SQLDatabaseChain.from_llm(
        llm, 
        db, 
        verbose=True, 
        return_intermediate_steps=True, 
        prompt=prompt, 
    )
    return db_chain
    
def get_stock_ticker(query):
    template = """You are a helpful assistant who extract company name from the human input.Please only output the company"""
    human_template = "{text}"
    llm = get_llm()

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])

    llm_chain = LLMChain(
        llm=llm,
        prompt=chat_prompt
    )

    company_name=llm_chain(query)['text'].strip()
    
    _DEFAULT_TEMPLATE = """Human: Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
<format>
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Result of SQLResult only"
</format>
Assistant: Understood, I will use the above format and only provide the answer.

Only use the following tables:
<tables>
CREATE TABLE stock_ticker (
	symbol text PRIMARY KEY,
	name text NOT NULL,
	currency text,
	stockExchange text, 
    exchangeShortName text
)
</tables>

If someone asks for the table stock ticker table, they really mean the stock_ticker table.
<examples>
Question: 
        What is the ticker symbol for Amazon in stock ticker table?
        Params: 
        Company name (name): Amazon
        
SQLQuery:SELECT symbol FROM stock_ticker WHERE name LIKE '%Amazon%'

</examples>

Question: \n\nHuman:{input} \n\nAssistant:

"""

    PROMPT = PromptTemplate(
        input_variables=["input", "dialect"], template=_DEFAULT_TEMPLATE
)
    db_chain = get_db_chain(PROMPT)

    company_ticker = db_chain("\n\nHuman: What is the ticker symbol for " + str(company_name) + " in stock ticker table? \n\nAssistant:")
    return company_name, company_ticker['result']

def get_stock_price(ticker, history=500):
    today = date.today()
    start_date = today - timedelta(days=history)
    data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
    return data

# Fetch top 5 google news for given company name
import re
import requests
def google_query(search_term):
    if "news" not in search_term:
        search_term=search_term+" stock news"
    url=f"https://www.google.com/search?q={search_term}"
    url=re.sub(r"\s","+",url)
    return url

from bs4 import BeautifulSoup
def get_recent_stock_news(company_name):
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'}

    g_query=google_query(company_name)
    res=requests.get(g_query,headers=headers).text
    soup=BeautifulSoup(res,"html.parser")
    news=[]
    for n in soup.find_all("div","n0jPhd ynAwRc tNxQIb nDgy9d"):
        news.append(n.text)
    for n in soup.find_all("div","IJl0Z"):
        news.append(n.text)

    if len(news)>6:
        news=news[:4]
    else:
        news=news
    news_string=""
    for i,n in enumerate(news):
        news_string+=f"{i}. {n}\n"
    top5_news="Recent News:\n\n"+news_string
    
    return top5_news

def stock_news_search(company_name):
    search=DuckDuckGoSearchRun()
    return search("Stock news about " + company_name)


# Get financial statements from Yahoo Finance
def get_financial_statements(ticker):
    if "." in ticker:
        ticker=ticker.split(".")[0]
    else:
        ticker=ticker
    company = yf.Ticker(ticker)
    balance_sheet = company.balance_sheet
    if balance_sheet.shape[1]>=3:
        balance_sheet=balance_sheet.iloc[:,:3]    # Only captures last 3 years of data
    balance_sheet=balance_sheet.dropna(how="any")
    balance_sheet = balance_sheet.to_string()
    return balance_sheet

from langchain.agents import load_tools
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain import LLMMathChain

tools=[
    Tool(
        name="get company ticker",
        func=get_stock_ticker,
        description="Get the company stock ticker"
    ),
    Tool(
        name="get stock data",
        func=get_stock_price,
        description="Use when you are asked to get stock data. This will output historic share price data. You should input the the stock ticker to it "
    ),
    Tool(
        name="get recent news",
        func=get_recent_stock_news,
        description="Use this to fetch recent news about stocks"
    ),

    Tool(
        name="get financial statements",
        func=get_financial_statements,
        description="Use this to get financial statement of the company. With the help of this data companys historic performance can be evaluaated. You should input stock ticker to it"
    ) 

]


from langchain.agents import initialize_agent 
def initializeAgent():
    agent=initialize_agent(
    llm=get_llm(),
    agent="zero-shot-react-description",
    tools=tools,
    verbose=True,
    max_iteration=2,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    output_key="output",
)
    return agent
    
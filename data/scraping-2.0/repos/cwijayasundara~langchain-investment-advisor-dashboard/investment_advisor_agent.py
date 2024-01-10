import os
import pinecone
import yfinance as yf
import investment_advisor_util
import pandas as pd
import streamlit as st

from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from yahooquery import Ticker
from datetime import datetime, timedelta
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from investment_advisor_util import stocks

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')  # find next to api key in console
PINECONE_ENV = os.getenv('PINECONE_ENV')  # find next to api key in console
index_name = 'semantic-search-openai'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

llm = ChatOpenAI(temperature=0.0)

# embedding model
embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=OPENAI_API_KEY
)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)
# connect to index assuming its already created
index = pinecone.Index(index_name)
print('Pinecone index status is', index.describe_index_stats())

text_field = "text"
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


def get_recommendation(question, stock_cik, expression):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vectorstore.as_retriever()
    )
    enhanced_question = f"{stock_cik} {question} {expression} "
    print(enhanced_question)
    result = qa.run(enhanced_question)
    return result.translate(str.maketrans("", "", "_*"))


def summerise_large_pdf_document(fileUrl):
    url = fileUrl
    loader = PyPDFLoader(url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
    return chain.run(texts)


# page construction
st.set_page_config(page_title="Relationship Manager Investment Dashboard ABC Plc", layout="wide",
                   initial_sidebar_state="collapsed", page_icon="agent.png")

col1, col2 = st.columns((1, 3))
icon = Image.open("agent.png")
col1.image(icon, width=100)

st.title("Relationship Manager Investment Dashboard ABC Plc")

selected_stock = col1.selectbox("Select a stock", options=list(stocks.keys()))

# Get stock data from yfinance
ticker = yf.Ticker(stocks[selected_stock]["symbol"])

# Calculate the date range for the last 365 days
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Get the closing prices for the selected stock in the last 365 days
data = ticker.history(start=start_date, end=end_date)
closing_prices = data["Close"]

# Plot the line chart in the first column
col1.line_chart(closing_prices, use_container_width=True)

# Get the company long description
long_description = ticker.info["longBusinessSummary"]

# Display the long description in a text box in the second column
col2.title("Company Overview")
col2.write(long_description)

# Use yahooquery to get earnings and revenue
ticker_yq = Ticker(stocks[selected_stock]["symbol"])
earnings = ticker_yq.earnings

financials_data = earnings[stocks[selected_stock]["symbol"]]['financialsChart']['yearly']

df_financials = pd.DataFrame(financials_data)
df_financials['revenue'] = df_financials['revenue']
df_financials['earnings'] = df_financials['earnings']
df_financials = df_financials.rename(columns={'earnings': 'yearly earnings', 'revenue': 'yearly revenue'})

numeric_cols = ['yearly earnings', 'yearly revenue']
df_financials[numeric_cols] = df_financials[numeric_cols].applymap(investment_advisor_util.format_large_number)
df_financials['date'] = df_financials['date'].astype(str)
df_financials.set_index('date', inplace=True)

# Display earnings and revenue in the first column
col1.write(df_financials)

summary_detail = ticker_yq.summary_detail[stocks[selected_stock]["symbol"]]

obj = yf.Ticker(stocks[selected_stock]["symbol"])

pe_ratio = '{0:.2f}'.format(summary_detail["trailingPE"])
price_to_sales = summary_detail["fiftyTwoWeekLow"]
target_price = summary_detail["fiftyTwoWeekHigh"]
market_cap = summary_detail["marketCap"]
ebitda = ticker.info["ebitda"]
tar = ticker.info["targetHighPrice"]
rec = ticker.info["recommendationKey"].upper()

# Format large numbers
market_cap = investment_advisor_util.format_large_number(market_cap)
ebitda = investment_advisor_util.format_large_number(ebitda)

# Create a dictionary for additional stock data
additional_data = {
    "P/E Ratio": pe_ratio,
    "52 Week Low": price_to_sales,
    "52 Week High": target_price,
    "Market Capitalisation": market_cap,
    "EBITDA": ebitda,
    "Price Target": tar,
    "Recommendation": rec
}

# Display additional stock data in the first column
for key, value in additional_data.items():
    col1.write(f"{key}: {value}")

col2.title("Opportunities for investors")

selected_stock_name = stocks[selected_stock]["name"]
selected_stock_url = stocks[selected_stock]["url"]

col2.subheader("Summary of the Last Quarter Financial Performance")
col2.write(summerise_large_pdf_document(selected_stock_url))

col2.subheader("Other Financial considerations")
col2.write(get_recommendation(selected_stock_name, "What are the key products and services of", "?"))
col2.write(get_recommendation(selected_stock_name,
                              "What are the new products and growth opportunities for", "?"))
col2.write(get_recommendation(
    selected_stock_name, "What are the key strengths of", "?"))
col2.write(
    get_recommendation(selected_stock_name, "Who are the key competitors of", "?"))
col2.write(get_recommendation(selected_stock_name, "What are the principal threats to", "?"))

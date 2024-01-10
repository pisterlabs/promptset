# imports
from bs4 import BeautifulSoup
from heapq import nlargest
import re
import nltk
import newsapi
from newspaper import Article
import json
import openai
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import PDFPlumberLoader
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import uuid
import pathlib
import io
import pdfplumber
import datetime
import datetime
from collections import deque
from datetime import date
import numpy as np
from plotly import graph_objs as go
import yfinance as yf
import requests
import pandas_ta as ta
import pandas as pd
from streamlit_extras.badges import badge as badge
import streamlit as st
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
YEAR = int(date.today().strftime("%Y"))
STARTOFYEAR = "f'{YEAR}-01-01'"


st.set_page_config(page_title='ESGParrot', page_icon=':parrot:', layout="centered",
                   initial_sidebar_state="auto", menu_items={
                       'Get Help': 'https://github.com/aidanaalund/predticker',
                       'Report a bug': "https://github.com/aidanaalund/predticker",
                       'About': "A stock dashboard with a focus on ESG ratings and analysis using NLP/LLM tools. Powered by NewsAPI, Yahoo! Finance, and OpenAI."
                   })

# Initialize session state keys
if 'stocks' not in st.session_state:
    st.session_state.stocks = set(["AAPL", "CAT", "TSLA", "MSFT"])
if 'predictiontext' not in st.session_state:
    st.session_state.predictiontext = ''
if 'currentlayoutbutton' not in st.session_state:
    st.session_state.currentlayoutbutton = None
if 'newsdictionary' not in st.session_state:
    st.session_state.newsdictionary = {}
if 'esgdictionary' not in st.session_state:
    st.session_state.esgdictionary = {}
if 'bbandcheck' not in st.session_state:
    st.session_state.bbandcheck = False
if 'volumecheck' not in st.session_state:
    st.session_state.volumecheck = False
if 'modelhistory' not in st.session_state:
    st.session_state.modelhistory = None
if 'currentstockmetadata' not in st.session_state:
    st.session_state.currentstockmetadata = None
if 'newgraph' not in st.session_state:
    st.session_state.newgraph = True
if 'currentdataframe' not in st.session_state:
    # make this set to what the selector is currently set to
    st.session_state.currentdataframe = None
if 'fileuploader' not in st.session_state:
    st.session_state.fileuploader = None
if 'pdftext' not in st.session_state:
    st.session_state.pdftext = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = {}
if 'apicounter' not in st.session_state:
    st.session_state.apicounter = 0
# User Input
col1, col2, col3 = st.columns([4, 3, 3])
with col1:
    st.title(":parrot: ESGParrot")
    st.markdown(
        "[Prices](#esgparrot) | [Metrics](#esg-statistics) | [ChatESG](#chatesg) | [News](#recent-news)")


# Adds an inputted stock string to a list of stocks in the state
# Checks for an invalid ticker by attempting to get the first value in a column


def addstock():
    if st.session_state.textinput:
        try:
            temp = yf.download(st.session_state.textinput, START, TODAY)
            test = temp['Close'].iloc[0]
            st.session_state.textinput = st.session_state.textinput.upper()
            st.session_state.stocks.add(st.session_state.textinput)
            st.session_state.selectbox = st.session_state.textinput
        except IndexError:
            st.error(
                body=f'Error: "{st.session_state.textinput}" is an invalid ticker.',
                icon='🚩')
            st.session_state.textinput = ''

# Sets a streamlit state boolean to true, making the graph render a new stock's data set.


def newgraph():
    st.session_state.newgraph = True


with col2:
    st.text('')
    selected_stock = st.selectbox("Select a ticker from your list:",
                                  st.session_state.stocks,
                                  key='selectbox',
                                  on_change=newgraph)
with col3:
    st.text('')
    newstock = st.text_input(label='Add a ticker to the list...',
                             placeholder="Type a ticker to add",
                             max_chars=4,
                             on_change=addstock,
                             key='textinput',
                             help='Please input a valid US ticker.')


# Load correctly formatted data in a pandas dataframe.


def add_indicators(df):
    df['SMA'] = ta.sma(df.Close, length=25)
    df['EMA12'] = ta.ema(df.Close, length=10)
    df['EMA26'] = ta.ema(df.Close, length=30)
    df['RSI'] = ta.rsi(df.Close, length=14)
    # returns multiple values
    # df['ADX'], df['DMP'], df['DMN'] = ta.adx(
    #     df.High, df.Low, df.Close, length=14)
    df['WILLR'] = ta.willr(df.High, df.Low, df.Close, length=25)
    # MACD stuff (without TALib!)
    macd = df['EMA26']-df['EMA12']
    macds = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macdh = macd - macds
    df['MACD'] = df.index.map(macd)
    df['MACDH'] = df.index.map(macdh)
    df['MACDS'] = df.index.map(macds)
    # Bollinger Bands
    df.ta.bbands(length=20, append=True)
    ta.bbands(df['Adj Close'], timeperiod=20)
    # Log return is defined as ln(d2/d1)
    # Starts at day 1
    df.ta.log_return(close=df['Adj Close'], cumulative=True, append=True)
    # Day over day log return
    df.ta.log_return(close=df['Adj Close'], cumulative=False, append=True)
    df['Target'] = np.where(df['LOGRET_1'] > 0, 1, 0)
    # df['return_'+benchmark] = 1
    # Percent return (now/day1)
    df.ta.percent_return(close=df['Adj Close'], cumulative=True, append=True)

    # Create signals
    df['EMA_12_EMA_26'] = np.where(df['EMA12'] > df['EMA26'], 1, -1)
    df['Close_EMA_12'] = np.where(df['Close'] > df['EMA12'], 1, -1)
    df['MACDS_MACD'] = np.where(df['MACDS'] > df['MACD'], 1, -1)

# Consider refactoring to use RapidAPI


@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    add_indicators(data)
    return data


data = load_data(selected_stock)
st.session_state.currentdataframe = data
# DATA PREPROCESSING
# grab first and last observations from df.date and make a continuous date range from that
dt_all = pd.date_range(
    start=data['Date'].iloc[0], end=data['Date'].iloc[-1], freq='D')
# check which dates from your source that also accur in the continuous date range
dt_obs = [d.strftime("%Y-%m-%d") for d in data['Date']]
# isolate missing timestamps
dt_breaks = [d for d in dt_all.strftime(
    "%Y-%m-%d").tolist() if not d in dt_obs]

# Define the plot types and the default layouts

candlestick = go.Candlestick(x=data['Date'], open=data['Open'],
                             high=data['High'], low=data['Low'],
                             close=data['Close'],
                             increasing_line_color='#2ca02c',
                             decreasing_line_color='#ff4b4b',
                             hoverinfo=None, name='Candlestick',)

volume = go.Scatter(x=data['Date'], y=data['Volume'])

bbu = go.Scatter(name='Upper Band', x=data['Date'], y=data['BBU_20_2.0'],
                 marker_color='rgba(30, 149, 242, 0.8)', opacity=.1,)
bbm = go.Scatter(name='Middle Band', x=data['Date'], y=data['BBM_20_2.0'],
                 marker_color='rgba(255, 213, 0, 0.8)', opacity=.8,)
bbl = go.Scatter(name='Lower Band', x=data['Date'], y=data['BBL_20_2.0'],
                 marker_color='rgba(30, 149, 242, 0.8)', opacity=.1,
                 fill='tonexty', fillcolor='rgba(0, 187, 255, 0.15)')

stocklayout = dict(

    yaxis=dict(fixedrange=False,
               ),
    xaxis_rangeslider_visible=False,
    xaxis=dict(
        fixedrange=False,
        rangebreaks=[
            dict(bounds=["sat", "mon"]),  # hide weekends
            dict(values=dt_breaks)
        ],

    ),
)


# Computes a scaled view for both plots based on the view mode and data
# Returned as an x range and two y ranges for each plot type (candle and volume)
header, subinfo = st.columns([2, 4])
change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
with header:
    price = data['Close'].iloc[-1]

    percentage = (float(data['Close'].iloc[-1] -
                        data['Close'].iloc[-2])/abs(data['Close'].iloc[-2]))*100.00

    st.metric(label=selected_stock,
              value='${:0.2f}'.format(price),
              delta='{:0.2f}'.format(change) +
              ' ({:0.2f}'.format(percentage)+'%) today'
              )
    recentclose = data['Date'].iloc[-1].strftime('%Y-%m-%d')
    st.caption(f'as of {recentclose}')
    st.markdown("""
        <style>
        [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
            gap: 0rem;
        }
        </style>
        """, unsafe_allow_html=True)

# TODO: This method returns slightly incorrect ranges for YTD


@st.cache_data
def defaultRanges(df, period):

    match period:
        case '1W':
            bf = 7
        case '1M':
            bf = 30
        case '6M':
            bf = 180
        case '1Y':
            bf = 365
        case '5Y':
            bf = 365*5
        case 'YTD':
            # TODO: the issue is that the first day of the year is new year's
            firstday = datetime.datetime(YEAR, 1, 1)
            # Try to get the first entry of the year
            # Then grab the data for closing and open
            # df.loc[start:end, 'Close']
            x = [firstday, df['Date'].iloc[-1]]
            ymax = df['High'].max()
            ymin = df['Low'].min()
            cbuffer = (ymax-ymin)*0.30
            ycandle = [ymin-cbuffer,
                       ymax+cbuffer]
            yvolume = [df['Volume'].min(), df['Volume'].max()]
            return x, ycandle, yvolume
        case 'Max':
            x = [df['Date'].iloc[0], df['Date'].iloc[-1]]
            ymax = df['High'].max()
            ymin = df['Low'].min()
            cbuffer = (ymax-ymin)*0.30
            ycandle = [ymin-cbuffer,
                       ymax+cbuffer]
            yvolume = [df['Volume'].min(), df['Volume'].max()]
            return x, ycandle, yvolume
        case _:
            bf = 30

    lower = df['Date'].iloc[-1]-np.timedelta64(bf, 'D')

    upper = df['Date'].iloc[-1]+np.timedelta64(1, 'D')

    x = [lower, upper]
    cbuffer = (df['High'].iloc[-bf: -1].max() -
               df['Low'].iloc[-bf:-1].min())*0.30
    ycandle = [df['Low'].iloc[-bf:-1].min()-cbuffer,
               df['High'].iloc[-bf:-1].max()+cbuffer]
    vbuffer = (df['Volume'].iloc[-bf:-1].max() -
               df['Volume'].iloc[-bf:-1].min())*0.30
    yvolume = [df['Volume'].iloc[-bf:-1].min()-vbuffer,
               df['Volume'].iloc[-bf:-1].max()+vbuffer]

    return x, ycandle, yvolume

# Calls defaultRanges to obtain proper scales and scales the plots passed in
# Assumes plot1 is a candlestick and plot2 is a volume/scatter plot


def scalePlots(df, period, plotly1, plotly2):
    xr, cr, vr = defaultRanges(df, period)
    plotly1.update_layout(xaxis_range=xr,
                          yaxis_range=cr,)
    plotly2.update_layout(xaxis_range=xr,
                          yaxis_range=vr,)


# Initialize candlestick and volume plots
fig = go.Figure(data=candlestick, layout=stocklayout)
fig2 = go.Figure(data=volume, layout=stocklayout)
if st.session_state.bbandcheck:
    fig.add_trace(bbm)
    fig.add_trace(bbu)
    fig.add_trace(bbl)
fig.update_layout(showlegend=False,
                  yaxis={'side': 'right'},
                  title='',
                  dragmode='pan',
                  yaxis_title='Share Price ($)',
                  modebar_remove=["autoScale2d", "autoscale", "lasso", "lasso2d",
                                  "resetview",
                                  "select2d",],
                  autosize=False,
                  width=700,
                  height=350,
                  margin=dict(
                      l=0,
                      r=0,
                      b=0,
                      t=0,
                      pad=0
                  ),)
# st.plotly_chart(fig, use_container_width=True)
fig2 = go.Figure(data=volume, layout=stocklayout)
fig2.update_layout(yaxis_title='Number of Shares',
                   autosize=False,
                   yaxis={'side': 'right'},
                   dragmode='pan',
                   modebar_remove=["autoScale2d", "autoscale", "lasso", "lasso2d",
                                   "resetview",
                                   "select2d",],
                   width=700,
                   height=400,
                   margin=dict(
                       l=0,
                       r=10,
                       b=0,
                       t=0,
                       pad=4
                   ),)


rangebutton = st.radio(
    label='Range Selector', options=('1W', '1M', '6M', 'YTD', '1Y', '5Y', 'Max'),
    horizontal=True, index=1, label_visibility='collapsed')

if st.session_state.newgraph:
    string = rangebutton
    scalePlots(st.session_state.currentdataframe, string, fig, fig2)
    st.session_state.newgraph = False


if rangebutton == '1W':
    scalePlots(st.session_state.currentdataframe, '1W', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '1M':
    scalePlots(st.session_state.currentdataframe, '1M', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '6M':
    scalePlots(st.session_state.currentdataframe, '6M', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == 'YTD':
    scalePlots(st.session_state.currentdataframe, 'YTD', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '1Y':
    scalePlots(st.session_state.currentdataframe, '1Y', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == '5Y':
    scalePlots(st.session_state.currentdataframe, '5Y', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)
elif rangebutton == 'Max':
    scalePlots(st.session_state.currentdataframe, 'Max', fig, fig2)
    st.plotly_chart(fig, use_container_width=True)
    if st.session_state.volumecheck:
        st.plotly_chart(fig2, use_container_width=True)


@st.cache_data(show_spinner=False)
def fetchInfo(ticker):
    ticker = yf.Ticker(ticker)
    info = ticker.get_info()
    return info


info = fetchInfo(selected_stock)
if info:
    if 'longName' in info:
        name = info['longName']
else:
    name = selected_stock
with st.expander(f"{name}'s summary"):
    if 'longBusinessSummary' in info:
        st.caption(info['longBusinessSummary'])
    else:
        st.error(f"{name}'s summary not available")


@st.cache_data(show_spinner=False)
def fetchESG(ticker):
    url = f"https://yahoo-finance127.p.rapidapi.com/esg-score/{ticker}"

    headers = {
        "RapidAPI-Key": st.secrets["yahoofinancekey"],
        "RapidAPI-Host": "yahoo-finance127.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)

    return response.json()


st.subheader("ESG Statistics:")
json = fetchESG(selected_stock)
if 'message' not in json:
    delta = 0
    url = "https://www.sustainalytics.com/corporate-solutions/esg-solutions/esg-risk-ratings"
    o, e, s, g = st.columns([2, 1, 1, 1])
    with o:
        value = json['totalEsg']['fmt']
        st.metric(label='Overall ESG Risk', value=value, delta=None,
                  help=f'{json["percentile"]["fmt"]}th percentile in the {json["peerGroup"]} peer group as of {json["ratingMonth"]}/{json["ratingYear"]}.')
        tier = float(value)
        tierstring = 'bug!'
        if tier > 0 and tier < 10:
            tierstring = 'Negligible'
            st.markdown(
                f'<h5 style="color:#d9d9d9;font-size:14px;">{"Negligible"}</h5>', unsafe_allow_html=True)
        elif tier >= 10 and tier < 20:
            tierstring = 'Low'
            st.markdown(
                f'<h5 style="color:#f4e7cc;font-size:14px;">{"Low"}</h5>', unsafe_allow_html=True)
        elif tier >= 20 and tier < 30:
            tierstring = 'Medium'
            st.markdown(
                f'<h5 style="color:#ffdca7;font-size:14px;">{"Medium"}</h5>', unsafe_allow_html=True)
        elif tier >= 30 and tier < 40:
            tierstring = 'High'
            st.markdown(
                f'<h5 style="color:#ffc46d;font-size:14px;">{"High"}</h5>', unsafe_allow_html=True)
        elif tier > 40:
            tierstring = 'Severe'
            st.markdown(
                f'<h5 style="color:#f89500;font-size:14px;">{"Severe"}</h5>', unsafe_allow_html=True)
    with e:
        st.metric(label='Environment Risk',
                  value=json['environmentScore']['fmt'], delta=None)
    with s:
        st.metric(label='Social Risk',
                  value=json['socialScore']['fmt'], delta=None)
    with g:
        st.metric(label='Governance Risk',
                  value=json['governanceScore']['fmt'], delta=None)
    graphic, info = st.columns([3, 2])
    with graphic:
        st.image(
            'https://raw.githubusercontent.com/aidanaalund/predticker/main/resources/sustainalytics_system.png')
    with info:
        st.caption(
            'Overall risk is calculated by adding each individual risk score. Higher ESG scores are generally related to higher valuation and less volatility. [Learn more](%s)' % url)

else:
    st.error(f'Sustainability data is currently not available for {name}')


# @st.cache_resource
# def esgBert():
#     return pipeline("text-classification", model="nbroad/ESG-BERT")


# def analysis():
#     pipe = esgBert()
#     st.session_state.esgdictionary[f'{selected_stock}'] = pipe(
#         st.session_state.report_input)

# BERT TOPIC MODELING
# st.text_area('Topic model a sustainability report/blurb:',
#              help='Using ESGBert, top ESG areas in the text are identified. Unexpected behavior will occur if text other than sustainability reports are inputted.',
#              placeholder='Put report text here...',
#              key='report_input',
#              on_change=analysis,
#              )

# if f'{selected_stock}' in st.session_state.esgdictionary:
#     response = st.session_state.esgdictionary[f'{selected_stock}']
#     topic = response[0]['label'].replace(
#         '_', ' ')
#     st.caption('Strongest Topic: '+topic)


@st.cache_data(show_spinner=False)
def findCsrLinks(company_name):
    # Construct the search query using the company name and keywords
    search_query = f"{company_name}"

    # Perform a web search using a search engine like Google
    search_url = f"https://www.google.com/search?q={search_query}"
    response = requests.get(search_url)

    if response.status_code == 200:
        # Parse the HTML content of the search results page
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the search result links
        search_results = soup.find_all('a')

        # Iterate through the search results to find relevant links
        csr_links = []
        for link in search_results:
            href = link.get('href')
            if href and ("sustainability" in href or 'impact' in href) and 'report' in href:
                sep = '&'
                stripped = href.split(sep, 1)[0]
                csr_links.append(stripped)
        if csr_links:
            return csr_links[0]
        else:
            return None
    else:
        print("Failed to fetch search results.")


company_name = f"{name} CSR Report"
csr_links = findCsrLinks(company_name)
if csr_links:
    st.subheader('Found Impact Reporting :page_facing_up::')
    st.caption(csr_links[7:])


# CHATBOT SECTION

# Left in for future tuning of .pdf reading
# @st.cache_data
# def parse():
#     if file is not None:
#         data = []
#         tables = []
#         with pdfplumber.open(file) as pdf:
#             pages = pdf.pages
#             # st.session_state.pdftext = len(pdf.pages)
#             # for p in pages:
#             #     data.append(p.extract_text())
#             # tables are returned as a list
#             # test for tesla 2021 report
#             # table = pages[67].extract_table()
#         st.session_state.pdftext = len(pages)


# TODO: determine how to cache embeddings and use hypothetical document embedding (HyDE).
# @st.cache_data(show_spinner=False)
def generateResponse(uploaded_file, openai_api_key, context, query_text, ticker):
    # Load document if file is uploaded
    if uploaded_file and openai_api_key and query_text != '':
        # create virtual file path for Langchain pdf reader interface
        filepath = pathlib.Path(str(uuid.uuid4()))
        filepath.write_bytes(uploaded_file.getvalue())

        fullquery = context+f'\n{query_text}'

        loader = PDFPlumberLoader(str(filepath))
        # Split documents into chunks
        documents = loader.load_and_split()
        # Select embeddings
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            # Create a vectorstore from documents
            db = Chroma.from_documents(documents, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()
            # Create QA chain
            qa = RetrievalQA.from_chain_type(llm=OpenAI(
                openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
            # get rid of temporary file path made
            if filepath.is_file():
                filepath.unlink()

            response = qa.run(fullquery)
        except:
            return None

        if str(ticker) in st.session_state.conversation:
            st.session_state.conversation[ticker].append(query_text)
            st.session_state.conversation[ticker].append(response)
            return response
        else:
            st.session_state.conversation[ticker] = []
            st.session_state.conversation[ticker].append(query_text)
            st.session_state.conversation[ticker].append(response)
            return response
    else:
        try:
            openai.api_key = openai_api_key
            qacontext = """Answer the question truthfully based on the text below. If
            the question does not seem to be related to sustainability or ESG 
            (Environmental, Social, and Governance) topics, do not answer the question and 
            state that you are not supposed to answer off topic questions."""
            fullquery = qacontext+f'\n{query_text}'
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"{fullquery}"},
                ]
            )
            if str(ticker) in st.session_state.conversation:
                st.session_state.conversation[ticker].append(query_text)
                st.session_state.conversation[ticker].append(
                    response['choices'][0]['message']['content'])
                return response['choices'][0]['message']['content']
            else:
                st.session_state.conversation[ticker] = []
                st.session_state.conversation[ticker].append(query_text)
                st.session_state.conversation[ticker].append(
                    response['choices'][0]['message']['content'])
                return response['choices'][0]['message']['content']
        except:
            return None


key = st.secrets['openapikey']
# the submit button will update all of the values inside the form all at once!
context = """Answer the question truthfully based on the text below. 
First answer the question, then include a verbatim quote with quote marks 
supporting your answer and a comment where to find it in the text (page number).
After the quote write a summary that explains your answer. Use bullet points."""

# one, two = st.columns([2, 2])
# with one:
st.subheader(
    'ChatESG :speaking_head_in_silhouette:: an LLM that understands ESG documents')
esgfile = st.file_uploader(label='Upload CSR report:',
                           type=['pdf'], help='PDF only.')
# st.info('Due to free OpenAI API access, ChatESG is designed to only handle 4 requests with a free API key.')

key = st.secrets['openapikey']
with st.chat_message(name="ChatESG", avatar='assistant'):
    st.write(
        'Hello! To make me work the best, look at my guide!')
    with st.expander('Usage Guide + Types of questions to ask me!'):
        st.markdown("""
        - Guide:
            - ChatESG has two modes, document analysis and general Q/A.
            The first mode runs when a document is uploaded, and the second mode
            runs if no document is submitted when a query is made.
            - If you query ChatESG with a document, ChatESG will try to find
            an answer and cite where it found the material.
            - If ChatESG is in Q/A mode, it will attempt to answer your question.
            - ChatESG does not currently remember previous inputs. Please account for this
            when interacting with it!
            - ChatESG may be incorrect on some issues or say it does not know an answer. 
            As a language model, it can often have a flawed understanding of topics. Always 
            verify what it cites and potential biases of the company who wrote the document.
            - Try to use specific keywords and language.
        - Question Ideas:
            - How does [company] plan to reduce their carbon emissions?
            - What is ESG?
            - How does [company] utilize diversity, equity, and inclusion in their business?
            - What is an area [company] can improve on in hitting emissions targets?
        """)

if selected_stock in st.session_state.conversation:
    user = True
    for message in st.session_state.conversation[selected_stock]:
        if user:
            with st.chat_message(name='User', avatar='user'):
                st.write(message)
            user = False
        else:
            with st.chat_message(name="ChatESG", avatar='assistant'):
                st.write(message)
            user = True


input = st.chat_input(
    placeholder="Ask a question about ESG or an impact report")

if input:
    if not st.session_state.apicounter >= 4:
        with st.chat_message(name='User', avatar='user'):
            st.write(input)
        with st.chat_message(name="ChatESG", avatar="assistant"):
            with st.spinner(text='Generating Response'):
                if st.session_state.userkey == '':
                    print('normal!')
                    msg = generateResponse(esgfile, key, context,
                                           input, selected_stock)
                else:
                    msg = generateResponse(esgfile, st.session_state.userkey, context,
                                           input, selected_stock)
            if msg is not None:
                st.write(msg)
            else:
                st.error(
                    'Invalid OpenAI API Key. Please enter a valid key or use the community key.')
    else:
        st.toast('Reached the maximum queries allowed for one user. Sorry!')

st.divider()

# News Section:
st.subheader('Recent News:')


@st.cache_data(show_spinner=False)
def fetchNews(name, endpoint):
    try:
        # TODO: make query parameter only get articles 'about' a company

        query_params = {
            'q': f'{name}',
            "sortBy": "relevancy",
            "apiKey": st.secrets["newsapikey"],
            "page": 1,
            "pageSize": 3,
            "language": "en",
            # can't mix sources with category
            # "sources": "reuters,cbs-news,the-washington-post,the-wall-street-journal,financial-times",
            # args for 'everything' endpoint only
            # "excludeDomains": "readwrite.com",
            # "domains": 'cnbc.com/business,usatoday.com/money/,cnn.com/business,gizmodo.com/tech,apnews.com/business,forbes.com/business/,bloomberg.com,newsweek.com/business,finance.yahoo.com/news/,',
            # args for top-headlines endpoint only
            # "category": "business"
        }
        if endpoint == "https://newsapi.org/v2/top-headlines":
            query_params["category"] = "business"
        else:
            query_params["excludeDomains"] = "readwrite.com"

        main_url = endpoint

        # fetching data in json format
        res = requests.get(main_url, params=query_params)
        response = res.json()

        # getting all articles
        return response

    except:
        st.error("News search failed.")


if f'{selected_stock}' not in st.session_state.newsdictionary:
    st.session_state.newsdictionary[f'{selected_stock}'] = fetchNews(
        name, "https://newsapi.org/v2/top-headlines")
    if st.session_state.newsdictionary[f'{selected_stock}']['totalResults'] == 0:
        st.session_state.newsdictionary[f'{selected_stock}'] = fetchNews(
            name, "https://newsapi.org/v2/everything")


@st.cache_resource
def sentimentModel():
    return pipeline("sentiment-analysis")


# create dropdowns for each article
if st.session_state.newsdictionary[f'{selected_stock}']['totalResults'] > 0:
    for ar in st.session_state.newsdictionary[f'{selected_stock}']['articles']:
        try:
            with st.expander(ar['title']):
                url = ar["url"]
                # fullarticle = Article(url)
                # fullarticle.download()
                # fullarticle.parse()
                # fullarticle.nlp()
                # data_df = fullarticle.keywords
                # print(type(data_df)) # list
                stripped = ar['publishedAt'].split("T", 1)[0]
                st.caption(f"{ar['description']}")
                # st.caption(f"{fullarticle.text}")
                sentbutton = st.button(label='Perform sentiment analysis...',
                                       key=url)
                if sentbutton:
                    sentiment_pipeline = sentimentModel()
                    # TODO: improve pipeline, and get full article contents
                    sent = sentiment_pipeline(ar['title'])
                    st.text(sent[0]['label'])
                st.caption(f'[Read at {ar["source"]["name"]}](%s)' % url)
                if ar["author"]:
                    st.caption(f'Written by {ar["author"]}')
                st.caption(f'{stripped}')
        except:
            st.error("Failed to grab article.")
else:
    st.error('No articles found.')

# Extras + Debug Menu
st.divider()
with st.expander('Extras & Settings Menu:'):
    st.caption(f"{name}'s dataframe:")
    st.dataframe(data=st.session_state.currentdataframe,
                 use_container_width=True)
    st.caption('ChatESG Settings:')
    userkey = st.text_input('Use your own OpenAI API Key:',
                            type='password', help="Please refer to OpenAI's website for pricing info.", key='userkey')
    st.caption('Graph display settings:')
    bbandcheck = st.checkbox(label="Display Bollinger bands",
                             key='bbandcheck')
    volumecheck = st.checkbox(label="Display volume plot",
                              key='volumecheck')

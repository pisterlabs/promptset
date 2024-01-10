import requests
import openai
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup


# use Streamlit's cache decorator to store the result of this function, so it only runs once
@st.cache_data
def get_news(ticker: str):
    # this function takes a ticker symbol as a string,
    # fetches the corresponding financial symbol with yfinance and returns the news related to this symbol
    symbol = yf.Ticker(ticker= ticker)
    return symbol.get_news()


# use Streamlit's cache decorator to store the result of this function, so it only runs once
@st.cache_data
def get_news_text(url: str):
    # this function takes a URL as a string,
    response = requests.get(url)
    # makes a GET request to fetch the web page,
    soup = BeautifulSoup(response.content, features='lxml')
    # parses the content with BeautifulSoup, and
    text = soup.find('div', attrs={'class': 'caas-body'}).text
    # returns the text of the news article
    return text


# use Streamlit's cache decorator to store the result of this function, so it only runs once
@st.cache_data
def get_news_summary(text: str, ticker: str):
    openai.api_key = st.secrets['openai']
    # this function takes a text string (a news article) and a ticker string, sends them to the OpenAI
    # GPT-3.5-turbo model to generate a summary, and returns the summary text
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Hello!"},
            {"role": "user", "content": "Please summarize the news article briefly. Also highlight the parts "
                                        f"that talk about {ticker}"
             },
            {"role": "user", "content": f"{text}"}
        ]
    )
    return response['choices'][0]['message']['content']


# Define a function to display and summarize company news
def company_news(ticker: str):
    # Get news related to the specified ticker
    news = get_news(ticker)
    # Set subheader for the news section
    st.subheader('I can summarize news for you!')
    # Create two columns in Streamlit interface
    col1, col2 = st.columns([1, 7])
    # Display the bot image in the first column
    col1.image('Resources/bot.jpeg')
    # Create a dropdown menu in the second column for selecting a news article to summarize
    selected_news = col2.selectbox('Select News title and I will create you a summary of that article',
                                   options=[title['title'] for title in news],
                                   )

    # Store the selected news URL in Streamlit's session state
    st.session_state.selected_news = [url['link'] for url in news if url['title'] == selected_news][0]

    # Add a button that triggers the news summary when clicked
    col2.button(label='**summarize!**', key='summarize_btn')

    # If the summarize button is clicked
    if st.session_state.summarize_btn:
        # Get the summary of the selected news
        st.session_state.news_summary = get_news_summary(text=st.session_state.selected_news,
                                                         ticker=st.session_state.selected_ticker)
        # Keep the summary expander open
        st.session_state.expanded = True

    # Initialize news_summary in the session state if it doesn't exist
    if 'news_summary' not in st.session_state:
        # Keep the summary expander closed
        st.session_state.expanded = False
        st.session_state.news_summary = ''

    # Create an expander for displaying the news summary
    with col2.expander(label='summary', expanded=st.session_state.expanded):
        # Display the news summary in the expander
        st.write(st.session_state.news_summary)

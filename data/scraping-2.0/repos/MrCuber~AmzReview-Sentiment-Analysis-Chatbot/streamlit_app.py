import openai
import time
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from chatbot import *
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from textblob import TextBlob
from urllib.request import urlopen
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient




# Configure your OpenAI API key
openai.api_key = st.secrets["YOUR_API_KEY"] #Enter you api key here

# Setting Streamlit Page UI

st.set_page_config(
    page_title="Product Helpers",
    page_icon="chart_with_upwards_trend"
)

# --------------------------------------------------------------------Sentiment Analysis----------------------------------------------------------------

def sentiment_analysis_page():
    st.subheader("Sentiment Analysis")
    product_url = st.text_input("Enter Amazon Product URL:")

    if st.button("Scrape Reviews and Perform Sentiment Analysis"):
        with st.spinner(text="Gathering Product Information..."):
            if product_url:
                reviews = scrape_page(product_url)
                if reviews:
                    # time.sleep(2)
                    st.spinner("Generating analysis...")
                    perform_sentiment_analysis(reviews)
                    st.success("Analysis completed successfully!")
                else:
                    st.error("Please check the provided URL. Failed to scrape reviews.")

def perform_sentiment_analysis(reviews):
    st.subheader("Sentiment Analysis Results")
    sentiments = []
    for review in reviews:
        blob = TextBlob(review)
        if blob.sentiment.polarity >= 0.5:
            sentiment = "Positive"
        else:
            if blob.sentiment.polarity <= -0.5:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
        sentiments.append(sentiment)

    sentiment_df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
    st.write(sentiment_df)

# --------------------------------------------------------------------Scraping Page Reviews--------------------------------------------------------------
        
def scrape_page(url):
    headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    client = ScrapingBeeClient(api_key=st.secrets["SCRAPER_API_KEY"])
    page = client.get(url)
    soup1 = BeautifulSoup(page.content, 'html.parser')
    soup2 = BeautifulSoup(soup1.prettify(), "html.parser")
 
    # Get product name
    product = soup2.find('span', {'id': 'productTitle'})
    if product:
        product_name = product.get_text().strip()
    else:
        product_name = ''

    # Get product category
    category_element = soup2.find('a', {'class': 'a-link-normal a-color-tertiary'})
    if category_element:
        category = category_element.get_text().strip()
    else:
        category = ''

    # Get product description
    description_element = soup2.find('div', {'name': 'description'})
    if description_element:
        description = description_element.get_text().strip()
    else:
        description = ''

    # Get product price
    price_element = soup2.find('span', 'a-offscreen')
    if price_element:
        price = price_element.get_text().strip()
    else:
        price = ''

    # Get product reviews
    reviews = []
    review_elements = soup2.find_all('span', {'class': 'a-size-base review-text'})
    for review_element in review_elements:
        reviews.append(review_element.get_text().strip())

    # Get product rating
    rating_element = soup2.find('span', {'class': 'a-icon-alt'})
    if rating_element:
        rating = rating_element.get_text().strip()
    else:
        rating = ''

    data = {
        'Product Name': [product_name],
        'Category': [category],
        'Description': [description],
        'Price': [price],
        'Reviews': ['\n'.join(reviews)],
        'Rating/Specifications': [rating]
    }
    df = pd.DataFrame(data)
    return reviews

# --------------------------------------------------------------------Main Function----------------------------------------------------------------------

def main():
    st.title("Amazon Review Sentiment Analysis and Chatbot")

    st.sidebar.header("Navigation Slider")
    # selected_page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Chatbot"])
    selected_page = st.sidebar.select_slider('Slide to select', options=['Sentiment Analysis','Chatbot'])
    if selected_page == "Sentiment Analysis":
        sentiment_analysis_page()
    elif selected_page == "Chatbot":
        st.subheader("Chatbot")
        user_input = st.text_input("Ask you query here:")
        if st.button("Chat"):
            if user_input.strip() != "":
                response = amz_chatbot(user_input)
                st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()

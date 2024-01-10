import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
import openai
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from chatbot import *
from bs4 import BeautifulSoup
from scrapingbee import ScrapingBeeClient

# Configure your OpenAI API key
openai.api_key = "xxxxx" #Enter you api key here


def perform_sentiment_analysis(reviews):
    st.subheader("Sentiment Analysis Results")
    sentiments = []

    for review in reviews:
        blob = TextBlob(review)
        sentiment = "Positive" if blob.sentiment.polarity >= 0.5 else ("Negative" if blob.sentiment.polarity <= -0.5 else "Neutral")
        sentiments.append(sentiment)

    sentiment_df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})
    st.write(sentiment_df)


        
def scrape_amazon_product_page(url):
    headers = {"User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
    client = ScrapingBeeClient(api_key='xxxxx')  #Enter your scraping bee api key here
    page = client.get(url)
    soup1 = BeautifulSoup(page.content, 'html.parser')
    soup2 = BeautifulSoup(soup1.prettify(), "html.parser")
 

    product = soup2.find('span', {'id': 'productTitle'})
    product_name = product.get_text().strip() if product else ''

    category_element = soup2.find('a', {'class': 'a-link-normal a-color-tertiary'})
    category = category_element.get_text().strip() if category_element else ''

    description_element = soup2.find('div', {'name': 'description'})
    description = description_element.get_text().strip() if description_element else ''

    price_element = soup2.find('span', 'a-offscreen')
    price = price_element.get_text().strip() if price_element else ''

    reviews = []
    review_elements = soup2.find_all('span', {'class': 'a-size-base review-text'})
    for review_element in review_elements:
        reviews.append(review_element.get_text().strip())

    rating_element = soup2.find('span', {'class': 'a-icon-alt'})
    rating = rating_element.get_text().strip() if rating_element else ''

    data = {
        'Product Name': [product_name],
        'Category': [category],
        'Description': [description],
        'Price': [price],
        'Reviews': ['\n'.join(reviews)],
        'Rating/Specifications': [rating]
    }
    df = pd.DataFrame(data)


    return reviews;


def main():
    st.title("Sentiment Analysis & Chatbot")

    st.sidebar.header("Navigation")
    selected_page = st.sidebar.radio("Go to", ["Sentiment Analysis", "Chatbot"])

    if selected_page == "Sentiment Analysis":
        sentiment_analysis_page()
    elif selected_page == "Chatbot":
        st.subheader("Chatbot")
        user_input = st.text_input("Ask a question or provide an inquiry:")
        if st.button("Chat"):
            if user_input.strip() != "":
                response = generate_chatbot_response(user_input)
                st.write("Chatbot: " + response)

def sentiment_analysis_page():
    st.subheader("Sentiment Analysis")
    product_url = st.text_input("Enter Amazon Product URL:")

    if st.button("Scrape Reviews and Perform Sentiment Analysis"):
        if product_url:
            reviews = scrape_amazon_product_page(product_url)  # Scrape reviews from the provided URL
            
            if reviews:
                perform_sentiment_analysis(reviews)  # Perform sentiment analysis on the scraped reviews
            else:
                st.warning("Failed to scrape reviews. Please check the provided URL.")

if __name__ == "__main__":
    main()

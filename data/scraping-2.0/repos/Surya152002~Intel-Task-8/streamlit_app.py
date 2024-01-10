import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os  # Added for file path handling
import openai


openai.api_key =  st.secrets["OPENAI_API_KEY"]




# Download the necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess data
def preprocess_data(df):
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df

def chatbot_response(user_input, predefined_responses):
    vectorizer = TfidfVectorizer(tokenizer=lambda text: nltk.word_tokenize(text, language='english'),
                                 stop_words=stopwords.words('english'))
    vectors = vectorizer.fit_transform([user_input] + predefined_responses)
    cosine_matrix = cosine_similarity(vectors)
    response_idx = np.argmax(cosine_matrix[0][1:])
    return predefined_responses[response_idx]

def get_gpt3_response(user_input):
    prompt = f"Answer questions related to cryptocurrency.\n\nUser: {user_input}\nBot:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50  
    )
    return response.choices[0].text

# Streamlit app
def main():
    st.title("Cryptocurrency Price Analysis Bot")
    csv_path = './crypto_dataset (9).csv'
    
    if not os.path.exists(csv_path):
        st.error("The specified CSV file does not exist. Please provide a valid file path.")
        return

    input_data = pd.read_csv(csv_path, parse_dates=['Timestamp'], index_col='Timestamp')
    input_data = preprocess_data(input_data)
    coins = ['BTC-USD Close', 'ETH-USD Close', 'LTC-USD Close']
    coin_choice = st.selectbox("Select a cryptocurrency", coins)

    # Chatbot live interaction
    st.header("Chat with Analysis Bot")
    user_message = st.text_input("You: ")
    predefined_responses = [
        "I am here to help with your cryptocurrency decisions.",
        "Can you specify your query?"
    ]

    if user_message:
        bot_reply = get_gpt3_response(user_message)
        st.write(f"Bot: {bot_reply}")

if __name__ == "__main__":
    main()

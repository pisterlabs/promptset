import streamlit_authenticator as stauth
import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import re
import yaml
import json
import requests
import os
from gensim.models import word2vec
from sklearn.manifold import TSNE
from yaml.loader import SafeLoader
from openai import OpenAI
from dotenv import load_dotenv


def build_open_ai_client() -> None:
    load_dotenv()
    global CLIENT
    CLIENT = OpenAI(
        api_key = os.getenv("OPEN_AI_API_KEY"),
        organization = os.getenv("OPEN_AI_ORGANIZATION_KEY")
    )

def determine_features(title: str, description: str) -> dict:
    if not CLIENT:
        raise Exception("OpenAI client not set. Please run set_open_ai_client() first.")
    
    response = CLIENT.chat.completions.create(
        model = "gpt-3.5-turbo",
        temperature = 1,
        messages = [
            {
                "role": "user",
                "content": r'''
                    I have a dataset with two columns, title and description. Based on these two informations, I can determine the features.
                    For example, if the title is: FYY Leather Case with Mirror for Samsung Galaxy S8 Plus, Leather Wallet Flip Folio Case with Mirror and Wrist Strap for Samsung Galaxy S8 Plus Black.
                    And the description is: Premium PU Leather Top quality. Made with Premium PU Leather. Receiver design. Accurate cut-out for receiver. Convenient to Answer the phone without open the case. Hand strap makes it easy to carry around. RFID Technique: Radio Frequency Identification technology, through radio signals to identify specific targets and to read and copy electronic data. Most Credit Cards, Debit Cards, ID Cards are set-in the RFID chip, the RFID reader can easily read the cards information within 10 feet(about 3m) without touching them. This case is designed to protect your cards information from stealing with blocking material of RFID shielding technology. 100% Handmade. Perfect craftsmanship and reinforced stitching make it even more durable. Sleek, practical, and elegant with a variety of dashing colors. Multiple Functions Card slots are designed for you to put your photo, debit card, credit card, or ID card while on the go. Unique design. Cosmetic Mirror inside made for your makeup and beauty. Perfect Viewing Angle. Kickstand function is convenient for movie-watching or video-chatting. Space amplification, convenient to unlock. Kickstand function is convenient for movie-watching or video-chatting.
                    the features will be: {
                        "category": "Phone Accessories",
                        "material": "Premium PU Leather",
                        "features": {
                        "receiver_design": "Accurate cut-out for receiver. Convenient to Answer the phone without opening the case.",
                        "hand_strap": "Yes",
                        "RFID_technique": "Protection of card information with RFID shielding technology",
                        "handmade": "100% Handmade",
                        "stitching": "Reinforced stitching",
                        "functions": {
                            "card_slots": "Yes",
                            "cosmetic_mirror": "Yes",
                            "kickstand_function": "Yes, convenient for movie-watching or video-chatting",
                            "space_amplification": "Yes, convenient to unlock"
                        },
                        "color_options": "Variety of dashing colors",
                        "compatibility": "Samsung Galaxy S8 Plus"
                        }
                    }
                '''
            },
            {
                "role": "user",
                "content": rf'''
                    Based on that, what will be the features of a column with the title {title} and description {description}.
                    Answer me using a JSON format, using just the keys category, meterial and features. Also, don't use any text other than the json.
                    As a last resort, if you don't know a specific value, just leave it as null. But please try your best.
                    '''
            }
        ]
    )
    return json.loads(response.choices[0].message.content)

@st.cache_data
def get_data() -> pd.DataFrame:
    product_search_corpus = requests \
                    .get("https://datasets-server.huggingface.co/first-rows?dataset=spacemanidol%2Fproduct-search-corpus&config=default&split=train") \
                    .json() \
                    ['rows']
    return pd.DataFrame((row['row'] for row in product_search_corpus))

@st.cache_data
def classify_data(data: pd.DataFrame) -> pd.DataFrame:
    for index, row in data.iterrows():
        for key, value in determine_features(title = row['title'], description = row['text']).items():
            if key not in data.columns:
                data[key] = None
            data.at[index, key] = str(value)
    return data

@st.cache_data
def build_model(data: pd.DataFrame, column: str) -> word2vec.Word2Vec:
    nltk.download('stopwords')
    STOP_WORDS = nltk.corpus.stopwords.words()
    regex = re.compile('([^\s\w]|_)+')

    data = data[['title', 'text', 'category', 'material', 'features']].dropna(how="any")
    data[column] = data[column].apply(lambda text: regex.sub('', text))
    corpus = [[token.lower() for token in str(sentence).split(" ") if token not in STOP_WORDS and token.isalpha()] for sentence in data[column]]
    return word2vec.Word2Vec(corpus, window=20, min_count=1, workers=4)

def tsne_plot(model) -> plt.Figure:
    labels = model.wv.index_to_key
    tokens = model.wv.vectors

    tsne_model = TSNE(perplexity=20, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = [value[0] for value in new_values]
    y = [value[1] for value in new_values]

    figure = plt.figure(figsize=(16, 16))
    plt.style.use('dark_background')
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(
            labels[i],
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom'
        )
        
    return figure

def BuildAuthenticator():
    with open('credentials.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    return stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )
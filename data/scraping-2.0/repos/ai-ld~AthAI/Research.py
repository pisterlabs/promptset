import streamlit as st
import pandas as pd
import openai
from datetime import datetime
import os
from apify_client import ApifyClient
from langchain import OpenAI
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from langchain.agents import create_pandas_dataframe_agent


# embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

openai.api_key = st.secrets["oai-key"]
client = ApifyClient(st.secrets["apify-key"])

encoding = tiktoken.get_encoding(embedding_encoding)

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def fetch_twitter_data(handles, n, from_date=None, to_date=None):
    run_input = {
        "handle": handles,
        "tweetsDesired": n,
        "profilesDesired": len(handles),
    }

    if from_date is not None:
        run_input["from_date"] = from_date
    if to_date is not None:
        run_input["to_date"] = to_date
    run = client.actor("quacker/twitter-scraper").call(run_input=run_input)
    tweet_data = [(item['created_at'], item['full_text'], item['user']['screen_name']) for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    df = pd.DataFrame(tweet_data, columns=['Date', 'Text', 'Author'])
    return df

def fetch_twitter_data_by_keyword(keyword, n, from_date=None, to_date=None):
    run_input = {
        "searchTerms": keyword,
        "tweetsDesired": n,
    }

    if from_date is not None:
        run_input["from_date"] = from_date
    if to_date is not None:
        run_input["to_date"] = to_date
    run = client.actor("quacker/twitter-scraper").call(run_input=run_input)
    tweet_data = [(item['created_at'], item['full_text'], item['user']['screen_name']) for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    df = pd.DataFrame(tweet_data, columns=['Date', 'Text', 'Author'])
    return df

def generate_embeddings(df):
    df['embedding'] = df.Text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
    embeddings = df['embedding'].apply(lambda x: np.array(x))
    embeddings_matrix = np.vstack(embeddings)
    return embeddings_matrix


def generate_summary(messages, model="gpt-3.5-turbo"):
    conversation = [{"role": "system", "content": "Summarize the following clusters in detail:"}]
    for message in messages:
        conversation.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()


def calculate_optimal_clusters(embeddings_matrix, max_k=4):
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings_matrix)

    k_values = list(range(2, max_k + 1))  # Start from 2 clusters, as silhouette_score requires at least 2
    silhouette_scores = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_embeddings)
        score = silhouette_score(scaled_embeddings, clusters)
        silhouette_scores.append(score)

    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k


def plot_clusters(embeddings_2d, clusters):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=clusters, palette='viridis', legend="full", s=100, alpha=0.7, ax=ax)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Cluster Visualization of OpenAI Embeddings')
    st.pyplot(fig)

def plot_similarity(df):
    df['Date'] = pd.to_datetime(df['Date'])
    fig, ax = plt.subplots()
    ax.scatter(df['Date'], df['Similarity'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity Over Time')
    st.pyplot(fig)

st.title("Political Campaign Analytics")

# Select the data source
data_source = st.selectbox("Select type", ["Twitter", "Polling (COMING SOON)", "Funding (COMING SOON)"])
st.session_state['fetch_data_pressed'] = False

if data_source == "Twitter":
    st.subheader("Twitter Analytics")
    # Input fields
    search_option = st.selectbox("Search by", ["Profile", "Keyword"])
    if search_option == "Profile":
        handles_input = st.text_input("Enter the Twitter handles (separated by commas)").replace(" ", "").strip("@")
        handles = handles_input.split(",")
    else:
        keyword = st.text_input("Enter the search keyword")
        keywords = keyword.split(",")

    n = st.number_input("Enter the number of tweets/posts to fetch:", min_value=1, max_value=1000, value=100, step=50)
    use_dates = st.checkbox("Use dates?")

    from_date = None
    to_date = None

    if use_dates:
        from_date = st.date_input("From date:")
        to_date = st.date_input("To date:")

    if st.button("Fetch Data"):
            if search_option == "Profile":
                st.write(f"Fetching {n} tweets for @{handles}")
                df = fetch_twitter_data(handles, n, from_date, to_date)
                #df = pd.read_csv("df.csv")
            else:
                st.write(f"Fetching {n} tweets containing '{keyword}'")
                df = fetch_twitter_data_by_keyword(keywords, n, from_date=from_date, to_date=to_date)

            st.write("Data fetched successfully.")

            st.write("Generating embeddings...")
            embeddings_matrix = generate_embeddings(df)
            st.write("Embeddings generated.")
            # Display text explanation for the selected analysis
            # Create expanders for each analysis type

            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')


            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='df.csv',
                mime='text/csv',
            )

            explanations = {
                "Clustering": """
                           Clustering analysis helps to identify groups of similar documents within a dataset.
                           In the context of a political campaign, this can be useful to identify common themes or patterns in public opinions.
                           For example, it could reveal groups of people sharing similar concerns or political views.
                           """,
                "Similarity Search": """
                           Similarity search enables you to find documents that are most similar to a given query.
                           In a political campaign, this can be helpful to find relevant documents or public opinions that match specific topics or talking points.
                           For example, you could use similarity search to find articles or social media posts discussing a particular policy proposal.
                           """,
                "Sentiment Analysis": """
                           Sentiment analysis is the process of determining the sentiment or emotion expressed in a piece of text.
                           For a political campaign, this can help to understand how people feel about certain issues or candidates.
                           For example, you could analyze sentiment in social media posts or news articles to gauge public opinion on a specific topic.
                           """
            }
            st.write(explanations["Clustering"])

            st.write("Clustering the data...")
            optimal_k = calculate_optimal_clusters(embeddings_matrix)
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embeddings_matrix)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(scaled_embeddings)

            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(scaled_embeddings)
            st.write("Visualizing the clusters...")
            plot_clusters(embeddings_2d, clusters)

            num_examples = 10  # Adjust this value based on your needs
            for cluster_idx in range(optimal_k):
                cluster_data = np.array(scaled_embeddings)[clusters == cluster_idx]
                distances = [np.linalg.norm(embedding - kmeans.cluster_centers_[cluster_idx]) for embedding in
                             cluster_data]
                closest_examples_idx = np.argsort(distances)[:num_examples]

                messages = [f"{df.loc[idx, 'Author']} + {df.loc[idx, 'Text']}" for idx in closest_examples_idx]

                st.write(f"\nCluster {cluster_idx}:")
                summary = generate_summary(messages)
                st.write(summary)

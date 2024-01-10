import requests
import streamlit as st
import pandas as pd
import numpy as np
import openai 
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams['font.family'] = 'AppleGothic'

def get_openai_response(prompt, api_key):
    if api_key:
        openai.api_key = api_key
    else: 
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message['content'].strip()

def get_cluster_description(cluster_words, api_key):
    prompt = f"이 클러스터는 {cluster_words}와 같은 단어들이 포함되어 있습니다. 이 클러스터의 주제와 내용을 한 문장으로 요약하세요."
    
    return get_openai_response(prompt, api_key)


def main():
    def load_data():
        df = pd.read_csv('/Users/marketdesigners/Documents/GitHub/Congress-Competition/model/legislation_data_with_embeddings.csv')
        df['Embedding'] = df['Embedding'].apply(ast.literal_eval)
        embeddings = np.array(df['Embedding'].tolist())
        kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
        df['Cluster'] = kmeans.labels_
        return df

    df = load_data()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    cluster_texts = [' '.join(df[df['Cluster'] == i]['legislative_context'].tolist()) for i in range(5)]
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_texts)
    feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    
    def get_top_words(tfidf_matrix, feature_names, cluster_num, top_n=10):
        cluster_row = tfidf_matrix[cluster_num].toarray().flatten()
        sorted_indices = np.argsort(cluster_row)[::-1]
        top_words = feature_names[sorted_indices[:top_n]].tolist()
        top_scores = cluster_row[sorted_indices[:top_n]]
        return top_words, top_scores
    
    st.title("Cluster Analysis with AI Description")

    api_key = st.text_input("Enter your OpenAI GPT API key:", type="password")

    if st.button('Show Cluster Visualization'):
        with st.expander("Cluster Visualization"):
            tsne = TSNE(n_components=2, random_state=0)
            reduced_embeddings = tsne.fit_transform(np.array(df['Embedding'].tolist()))
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['Cluster'])
            plt.colorbar(scatter)
            st.pyplot(fig)

    if st.button('Generate Cluster Descriptions'):
        for i in range(5):
            cluster_words = ', '.join(get_top_words(tfidf_matrix, feature_names, i, 10)[0])
            description = get_cluster_description(cluster_words, api_key)
            st.markdown(f"**Cluster {i} Description:** {description}")

    if st.button('Show Word Clouds for Each Cluster'):
        st.subheader("Word Clouds for Each Cluster")
        for i in range(5):
            cluster_text = ' '.join(df[df['Cluster'] == i]['legislative_context'].tolist())
            wordcloud = WordCloud(font_path='/System/Library/Fonts/Supplemental/AppleGothic.ttf', background_color='white').generate(cluster_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

    if st.button('Show Top Words and Descriptions in Each Cluster'):
        with st.expander("Top Words and Descriptions in Each Cluster"):
            for i in range(5):
                top_words, _ = get_top_words(tfidf_matrix, feature_names, i, 10)
                formatted_words = ', '.join(top_words)
                st.markdown(f"**Cluster {i}**: {formatted_words}")
                if api_key:
                    description = get_cluster_description(formatted_words, api_key)
                    st.markdown(f"**Description:** {description}")
                else:
                    st.markdown("**Description:** Enter your API key to generate descriptions.")

    if st.button('Show Related Laws for Each Cluster'):
        with st.expander("Related Laws for Each Cluster"):
            for i in range(5):
                cluster_df = df[df['Cluster'] == i]
                st.markdown(f"#### Cluster {i} Related Laws:")
                st.dataframe(cluster_df[['user_problem', 'legislative_context']])
                
if __name__ == "__main__":
    main()

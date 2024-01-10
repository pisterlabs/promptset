import os
import openai
# from openai.embeddings_utils import cosine_similarity
from sklearn.manifold import TSNE
import streamlit as st
from matplotlib import cm
import pandas as pd
import numpy as np
from ast import literal_eval
import nomic
from nomic import atlas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from sentence_transformers import SentenceTransformer, util
import torch


st.set_page_config(page_icon="🤖", layout="wide")
st.markdown("<h2 style='text-align: center;'>InstructLens: A Toolkit for Visualizing Instructions via Aggregated Semantic and Linguistic Rules</h2>", unsafe_allow_html=True)

def main():
    # Change the file from json to csv to make the preprocessing easier
    # datafile_path = "./alpaca_data.json"
    # with open(datafile_path, "r") as f:
    #     data = json.load(f)
    # df = pd.DataFrame(data)

    # Load csv file
    csv_file = "./alpaca_data.csv"
    df = pd.read_csv(csv_file, usecols=['instruction', 'input', 'output'])


    # Display the total number of documents
    st.markdown("""""")
    st.markdown("""""")
    st.markdown("""""")
    st.subheader("Basic Information about the Dataset")
    total_documents = len(df)
    st.write(f"Total number of documents: **{total_documents:,}**")
    st.divider()


    # Display the data
    # st.subheader("Data:")
    # st.write(df)
    # st.divider()
    

    # Display most similar 5 sentences
    st.subheader("Search similarity")
    col1, col2 = st.columns([3, 2])  # Adjust the ratio as needed
    with col1:
        # Search form in the left column
        form = st.form('Embeddings')
        question = form.text_input("Enter a sentence to search for semantic similarity", 
                                   value="How can we reduce air pollution?")
        num_sentences = form.number_input("Number of similar sentences to display", min_value=1, max_value=total_documents, value=5)
        btn = form.form_submit_button("Run")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    # csv_file_with_embeddings = "./alpaca_data_with_embeddings.csv"
    # df = pd.read_csv(csv_file_with_embeddings, converters={'embeddings': literal_eval})

    if btn:
        with col1: 
            with st.spinner("Searching for similar sentences..."):
                # Compute embedding for the input question
                question_embedding = model.encode(question, convert_to_tensor=True)

                # Compute embeddings for all sentences in the DataFrame
                # sentence_embeddings = model.encode(df['instruction'].tolist(), convert_to_tensor=True)
                saved_embeddings_df = pd.read_csv("sentence_embeddings.csv", converters={'embedding': literal_eval})
                sentence_embeddings = torch.tensor(saved_embeddings_df['embedding'].tolist())

                # Compute cosine similarities
                similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings)[0]

                # Convert similarities to numpy array for sorting
                similarities = similarities.cpu().numpy()

                # Get the top 5 most similar sentence indices
                top_indices = np.argsort(similarities)[::-1][1:num_sentences+1]

                # Display top 5 similar sentences
                st.write("Top 5 similar sentences:")
                for idx in top_indices:
                    st.write(f"Instruction: {df.iloc[idx]['instruction']}")
                    st.write(f"Output: {df.iloc[idx]['output']}")
                    st.write(f"Similarity: {similarities[idx]:.4f}")
                    st.write("---------")
                
                # Combine all embeddings for t-SNE (question embedding + sentence embeddings)
                all_embeddings = np.vstack([question_embedding.cpu().numpy(), sentence_embeddings.cpu().numpy()])

                # Perform t-SNE
                tsne = TSNE(n_components=2, random_state=0)
                embeddings_2d = tsne.fit_transform(all_embeddings)

                # Visualization
                marker_size = 1.5
                fig, ax = plt.subplots()
                colors = ['red'] + ['blue'] * len(sentence_embeddings)  # Red for query, blue for sentences
                ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, s=marker_size)

                # Highlight the question point
                ax.scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], c='green', edgecolors='black', label='Query', s=10)

                # Optional: Annotate points with their index or text (might be cluttered)
                # for i, txt in enumerate(df['instruction']):
                #     ax.annotate(i, (embeddings_2d[i+1, 0], embeddings_2d[i+1, 1]))

                ax.legend()
                ax.grid(False)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
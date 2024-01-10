import numpy as np
import pandas as pd
import polars as pl
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
from datetime import datetime
from ast import literal_eval
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info('Loading model from HuggingFace Hub...')
# Define the device (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Loading model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# Move the model to the specified device
model.to(device)
logger.info('Model loaded.')


def merge_phrases(df, n_from, n_to):
    # Group by linkedid and apply a lambda function to join the phrases from n_from to n_to
    merged = df.groupby('linkedid').apply(
        lambda x: ' '.join(x['text'].iloc[n_from:n_to]) if len(x['text']) >= n_to else ''
    )
    # Convert the result to a DataFrame and reset the index
    merged_df = pd.DataFrame(merged).reset_index()
    # Rename the columns
    merged_df.columns = ['linkedid', 'text']
    return merged_df


# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embeddings(df):
    # Sorting values
    top_n = 1000
    df = df.sort_values("linkedid").tail(top_n * 2)

    # Ensure that all entries in 'text' column are strings
    sentences = df['text'].astype(str).tolist()

    # Create DataLoader for batching
    batch_size = 32  # Adjust based on your GPU's memory
    dataloader = DataLoader(sentences, batch_size=batch_size, shuffle=False)

    embeddings_list = []
    
    for batch in dataloader:
        # Tokenize sentences
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

        # Move the encoded input to the specified device
        encoded_input = encoded_input.to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Apply mean pooling to get sentence embeddings
        attention_mask = encoded_input['attention_mask']
        embeddings = mean_pooling(model_output, attention_mask).cpu().numpy()
        embeddings_list.extend(embeddings)

    # Add embeddings to the DataFrame
    df['embedding'] = embeddings_list
    df = df.tail(top_n)  # Keep only the top_n entries

    return df


def clustering(df, n_clusters=4):
    # df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
    matrix = np.vstack(df.embedding.values)
    logger.info(f"Matrix shape: {matrix.shape}")

    # 1. Find the clusters using K-means
    # We show the simplest use of K-means. 
    # You can pick the number of clusters that fits your use case best.
    # n_clusters = 4
    logger.info(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    df["Cluster"] = labels

    # df.groupby("Cluster").Score.mean().sort_values()    
    return df, matrix


# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

def plot_clusters(df, matrix, legend_append_values=None):
    logger.info("Plotting clusters...")
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    x = [x for x, y in vis_dims2]
    y = [y for x, y in vis_dims2]
    
    colors = ["purple", "green", "red", "blue", "orange", "yellow", "pink", "brown", "gray", "black", "cyan", "magenta"]
    colors = colors[:len(np.unique(df.Cluster))]
    cluster_sizes = df.Cluster.value_counts(normalize=True).sort_values(ascending=False)
    
    # Initialize subplot
    fig = make_subplots(rows=1, cols=1)
    
    for category in cluster_sizes.index:
        color = colors[category]
        xs = np.array(x)[df.Cluster == category]
        ys = np.array(y)[df.Cluster == category]
        texts = df[df.Cluster == category]['text'].values  # Get the text for each point in this cluster

        cluster_percentage = cluster_sizes[category] * 100  # cluster_sizes is already normalized

         # Append values to the legend
        if legend_append_values is not None:
            legend_append = f', {legend_append_values[category]}'
        else:
            legend_append = ''

        # Add scatter plot to subplot
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, 
                mode='markers',
                marker=dict(color=color, size=5),
                hovertext=texts,  # Display the text when hovering over a point
                hoverinfo='text',  # Show only the hovertext
                name=f'Cluster {category} ({cluster_percentage:.2f}%)' + legend_append,
            )
        )

        avg_x = xs.mean()
        avg_y = ys.mean()

        # Add marker for average point to subplot
        fig.add_trace(
            go.Scatter(
                x=[avg_x], y=[avg_y],
                mode='markers',
                marker=dict(color=color, size=10, symbol='x'),
                name=f'Avg Cluster {category}',
                hoverinfo='name'
            )
        )

    fig.update_layout(showlegend=True, title_text="Clusters identified visualized in language 2d using t-SNE")
    fig.show()


def topic_samples_central(df, matrix, openai_key, n_clusters, rev_per_cluster):
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)
    # logger.info("Summarizing topics...")

    openai.api_key = openai_key
    topics = {}
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    # Apply t-SNE to obtain 2D coordinates for each data point
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
    vis_dims2 = tsne.fit_transform(matrix)

    line = '#'*3
    logger.info('Topics request:')

    # Gather samples from each cluster and add to the messages list.
    for i in range(n_clusters):
        # Filter the dataframe to only include data points in the current cluster
        # cluster_df = df[df.Cluster == i]
        cluster_df = df[df.Cluster == i].reset_index(drop=True)

        # Calculate the 2D center of the current cluster
        cluster_center = vis_dims2[cluster_df.index].mean(axis=0)

        # Calculate the Euclidean distance from each data point to the cluster center
        distances = np.sqrt(((vis_dims2[cluster_df.index] - cluster_center)**2).sum(axis=1))

        # Get the indices of the data points with the smallest distances
        closest_indices = distances.argsort()[:rev_per_cluster]

        # Get the corresponding reviews
        closest_reviews = cluster_df.iloc[closest_indices].text

        # Join the reviews with formatting
        reviews = "\n ".join(
            closest_reviews
            .str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .values
        )

        messages.append({"role": "user", "content": f"\nКластер {i} {cluster_center}:\n{reviews}"})
        logger.info(messages[-1]['content'])

    # Add the question asking for topic summaries.
    messages.append({"role": "user", "content": "Это фрагменты диалогов, восстановленных из разговоров колл центра. Эти фрагменты разговоров уже разделены на кластеры. Пожалуйста, дайте описание каждому кластеру так, что бы было ясно что его выделяет среди других кластеров. Ответ представьте в виде JSON структуры: {'Кластер 0': 'Тема 0', 'Кластер 1': 'Топик 1'}"})
    
    
    # Make the API call.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )

    # Assuming the response['choices'][0]['message']['content'] returns the JSON as string
    topics_json = response['choices'][0]['message']['content']
    # Log total tokens
    logger.info('total_tokens: '+str(response['usage']['total_tokens']))

    # Load the string as a dictionary
    topics_dict = json.loads(topics_json)

    # Get the list of topics.
    topics = list(topics_dict.values())

    # Log the topics.
    logger.info('Topics result:')
    logger.info(topics)

    return topics


# Defining a custom function to convert the string representation to a NumPy array
def convert_to_array(embedding_str):
    # Removing the square brackets and splitting the string by space to get the individual elements
    elements = embedding_str[1:-1].split()
    # Converting the elements to floats and creating a NumPy array
    return np.array([float(e) for e in elements])
    

def main():
    openai_key = input('Enter OpenAI key: ')
    dataset_path = '../../datasets/transcribations/transcribations_2023-04-27 16:07:39_2023-07-25 19:03:21_polars.csv'
    n_clusters = 4
    # Load calls and format to conversations
    # df = calls_to_converations(dataset_path, '2023-07-21', n_from=1, n_to=5)
    # df.to_csv('conversations.csv')
    
    # Load conversations
    df = pd.read_csv('conversations.csv')
    
    # Ada v2	$0.0001 / 1K tokens
    
    # df = get_embeddings(df, openai_key=openai_key)
    df = get_sentence_embeddings(df)
    df.to_csv("embeddings.csv")
    # Load embeddings
    # df = pd.read_csv('embeddings.csv')
    # df = pd.read_csv('local_conversations_embeddings.csv')

    # Reloading the original DataFrame from the CSV file
    # df = pd.read_csv('embeddings.csv')
    # Applying the custom conversion function to the 'embedding' column
    # df['embedding'] = df['embedding'].apply(convert_to_array)
    # Clustering
    df, matrix = clustering(df, n_clusters=n_clusters)

    # Summarize topics
    legend = topic_samples_central(df, matrix, openai_key=openai_key, n_clusters=n_clusters, rev_per_cluster=10)
    # Fake legend
    # legend = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3']
    # Plot clusters
    plot_clusters(df, matrix, legend_append_values=legend)

    logger.info('Done.')


if __name__ == "__main__":
    main()

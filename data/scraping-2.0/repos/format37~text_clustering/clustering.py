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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def calls_to_converations(data_path, date_pick, n_from, n_to):
    # data_path = '../../datasets/transcribations/transcribations_2023-02-03 07:01:26_2023-04-27 16:07:39_polars.csv'
    logger.info('Loading data from polars...')
    # Increase infer_schema_length
    df = pl.read_csv(data_path, infer_schema_length=100000)
    columns_to_keep = ['transcribation_date', 'side', 'start', 'text', 'linkedid']
    columns_to_drop = [col for col in df.columns if col not in columns_to_keep]
    df = df.drop(columns_to_drop)
    logger.info('Cropping data...')
    mask = df["transcribation_date"].apply(lambda x: x[:10]==date_pick)
    df = df.filter(mask)
    # Convert to pandas
    logger.info('Converting to pandas...')
    df = df.to_pandas()
    # Convert date column
    logger.info('Converting date column...')
    # df['transcribation_date'] = pd.to_datetime(df['transcribation_date'], format='%Y-%m-%d %H:%M:%S')
    # 2023-07-21T06:01:37.000000000
    df['transcribation_date'] = pd.to_datetime(df['transcribation_date'], format='%Y-%m-%dT%H:%M:%S.%f')
    # df['transcribation_date'] = df['transcribation_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    # Convert date column
    # df['transcribation_date'] = pd.to_datetime(df['transcribation_date'], format='%Y-%m-%d %H:%M:%S')
    # Create a boolean mask for the date range
    # mask = (df['transcribation_date'] >= '2023-02-04 12:00:00') & (df['transcribation_date'] <= '2023-02-04 13:00:00')
    # Apply the mask to the DataFrame
    # df = df.loc[mask]
    # Drop the rows with NaN values
    df.dropna(inplace=True)
    # Convert 'linkedid' to int64 type to remove scientific notation
    # df['linkedid'] = df['linkedid'].astype('int64')
    # Convert to string
    df['linkedid'] = df['linkedid'].astype(str)

    # Ensure that all entries in 'text' column are strings
    df['text'] = df['text'].astype(str)
    
    logger.info('Merging text sequences from '+str(len(df))+' rows...')
    # Merge texts
    # Sort the dataframe by 'transcribation_date', 'linkedid', and 'start'
    df.sort_values(by=['linkedid', 'start'], inplace=True)    

    # Iterate over each row in the dataframe
    # Initialize empty dataframe to store the merged rows
    merged_df = pd.DataFrame(columns=df.columns)

    # Initialize variables
    previous_linkedid = None
    previous_side = None
    merged_row = None

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Check if this row has the same speaker (side) and linkedid as the previous one
        if row['linkedid'] == previous_linkedid and row['side'] == previous_side:
            merged_row['text'] = str(merged_row['text']) + ' ' + row['text']
        else:
            # If this row has a different speaker or linkedid, save the previous merged row to the new dataframe
            if merged_row is not None:
                merged_df = pd.concat([merged_df, pd.DataFrame([merged_row])], ignore_index=True)
            # Start a new merged row with this row's data
            merged_row = row.copy()
        # Update the previous speaker and linkedid for the next iteration
        previous_side = row['side']
        previous_linkedid = row['linkedid']
    # Append the last merged row to the new dataframe
    if merged_row is not None:
        merged_df = pd.concat([merged_df, pd.DataFrame([merged_row])], ignore_index=True)
    logger.info('Merged to '+str(len(merged_df))+' rows.')
    logger.info('Cropping to N phrases...')
    # Crop to a conversation
    # Add '\n- ' before each phrase to make the conversation more readable
    merged_df['text'] = '- ' + merged_df['text']+ '\n'
    
    # Drop rows where linkedid count is less than n_to
    merged_df = merged_df[merged_df.groupby('linkedid')['linkedid'].transform('count') >= n_to]
    merged_df = merge_phrases(merged_df, n_from, n_to)
    # Remove empty rows
    logger.info('Cropped to '+str(len(merged_df))+' rows.')
    return merged_df


def get_embeddings(df, openai_key):
    # 1. Load the dataset
    # embedding model parameters
    embedding_model = "text-embedding-ada-002"
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
    max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

    # subsample to 1000 most recent text and remove samples that are too long
    top_n = 1000
    df = df.sort_values("linkedid").tail(top_n * 2)  # first cut to first 2k entries, assuming less than half will be filtered out
    df.drop("linkedid", axis=1, inplace=True)

    # Ensure that all entries in 'text' column are strings
    df['text'] = df['text'].astype(str)

    encoding = tiktoken.get_encoding(embedding_encoding)

    # omit reviews that are too long to embed
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens].tail(top_n)
    logger.info(f"Number of samples: {len(df)}")

    # 2. Get embeddings and save them for future reuse
    openai.api_key = openai_key
    # Ensure you have your API key set in your environment per the README: https://github.com/openai/openai-python#usage
    logger.info("Getting embeddings. This may take a few minutes...")
    df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=embedding_model))
    logger.info("Embeddings retrieved.")
    return df


def clustering(df, n_clusters=4):
    df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)  # convert string to numpy array
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
        cluster_df = df[df.Cluster == i]

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



def topic_samples_random(df, matrix, openai_key, n_clusters, rev_per_cluster):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Summarizing topics...")

    openai.api_key = openai_key
    topics = {}
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    # Gather samples from each cluster and add to the messages list.
    for i in range(n_clusters):
        reviews = "\n - ".join(
            df[df.Cluster == i]
            .text.str.replace("Title: ", "")
            .str.replace("\n\nContent: ", ":  ")
            .sample(rev_per_cluster, random_state=42)
            .values
        )
        messages.append({"role": "user", "content": f"Кластер {i}: \n - {reviews}"})

    # Add the question asking for topic summaries.
    # messages.append({"role": "user", "content": "Please, invent a topic for each cluster, accounting the differences between clusters. Represent the answer as a json structure: {'cluster 0': 'topic 0', 'cluster 1': 'topic 1'}"})
    messages.append({"role": "user", "content": "Пожалуйста, придумайте тему для каждого кластера, учитывая разницу между кластерами. Ответ представьте в виде JSON структуры: {'Кластер 0': 'Тема 0', 'Кластер 1': 'Топик 1'}"})

    # Make the API call.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )
    # Assuming the response['choices'][0]['message']['content'] returns the JSON as string
    topics_json = response['choices'][0]['message']['content']

    # Load the string as a dictionary
    topics_dict = json.loads(topics_json)

    # Get the list of topics.
    topics = list(topics_dict.values())

    # Log the topics.
    logger.info(topics)

    return topics
    

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
    
    df = get_embeddings(df, openai_key=openai_key)
    df.to_csv("embeddings.csv")
    # Load embeddings
    # df = pd.read_csv('embeddings.csv')
    # df = pd.read_csv('local_conversations_embeddings.csv')
    
    # Clustering
    df, matrix = clustering(df, n_clusters=n_clusters)

    # Summarize topics
    legend = topic_samples_central(df, matrix, openai_key=openai_key, n_clusters=n_clusters, rev_per_cluster=10)
    # Plot clusters
    plot_clusters(df, matrix, legend_append_values=legend)

    logger.info('Done.')


if __name__ == "__main__":
    main()

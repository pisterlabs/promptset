import streamlit as st
import json
import pickle
import numpy as np
from sklearn.cluster import KMeans
import openai
import matplotlib.pyplot as plt


def main():
    st.write("## Clustering w/Augmentation")
    # Load Twilio customer purchase data from a JSON file
    with open("customer_data.json", "r") as file:
        customer_data = json.load(file)

    # Create an embedding cache dictionary or load from a pickle file if it exists
    embedding_cache_file = "embedding_cache.pkl"

    try:
        with open(embedding_cache_file, "rb") as cache_file:
            embedding_cache = pickle.load(cache_file)
    except FileNotFoundError:
        embedding_cache = {}

    # Extract customer purchase features
    customer_embeddings = []
    for customer in customer_data:
        text_data = f"Customer {customer['customer_id']} has purchased {', '.join(customer['product_skus'])} products, uses {customer['programming_language']} programming language, and has an average monthly spend of ${customer['average_monthly_spend']}."

        # Check if embedding exists in the cache
        if text_data in embedding_cache:
            embedding = embedding_cache[text_data]
        else:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=text_data,
                temperature=0,
                max_tokens=64,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            embedding = response["choices"][0]["text"].replace("\n", "")
            embedding_cache[text_data] = embedding

        customer_embeddings.append(embedding)

    # Convert embeddings to NumPy array
    customer_embeddings = np.array(customer_embeddings)

    # Save the embedding cache to a pickle file
    with open(embedding_cache_file, "wb") as cache_file:
        pickle.dump(embedding_cache, cache_file)

    # Perform clustering using K-means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(customer_embeddings)
    labels = kmeans.labels_

    # Assign cluster labels to customers
    for i, customer in enumerate(customer_data):
        customer['cluster'] = labels[i]

    rev_per_cluster = 5

    # Define the clusters and themes
    clusters = []
    themes = []
    st.write("One way to use embeddings is to cluster similar embeddings together and then find a theme for each cluster. Let's try this with the customer data.")
    st.write("### Step 1: Get Customer Data")
    st.write(customer_data)
    st.write("### Step 2: Create Clusters")
    st.write("With the customer data we can create clusters using K-means clustering. We will use the embeddings of the customer data to create the clusters.")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        customer_embeddings[:, 0], customer_embeddings[:, 1], c=labels, cmap='viridis')

    # add colorbar to the plot
    cbar = plt.colorbar(scatter)

    # customize plot labels and title
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Customer Clusters")
    st.pyplot(fig)
    st.write("### Step 3: Augment to Get Themes")
    st.write("Now that we have clusters, we can get themes for each cluster. We will use the embeddings of the customer data to get the themes.")
    st.write("**Augmented Query**")
    augmented_query = """What do the following customer data have in common?\n\nCustomer data:\n{customer_data}\nRespond with a theme based on the product_sku, programming language and/or spend. Be specific about a theme with a maxium of three words, pick something specifc to the data you are reviewing based on the data of product_sku "sms", "voice", "flex", "segment", "sendgrid", "sms", "sms"] and or programming language:"""
    st.code(augmented_query)
    for i in range(n_clusters):
        clusters.append(i)

        cluster_customers = [
            customer for customer in customer_data if customer.get('cluster') == i]

        reviews = "\n".join(
            [f"Customer {customer['customer_id']}: {customer['product_skus']}, {customer['programming_language']}, ${customer['average_monthly_spend']}"
             for customer in cluster_customers]
        )

        # Check if theme analysis exists in the cache
        if reviews in embedding_cache:
            theme = embedding_cache[reviews]
            themes.append(theme)
        else:
            # remove themes from embedding cache
            embedding_cache.pop("reviews", None)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f'What do the following customer data have in common?\n\nCustomer data:\n"""\n{reviews}\n"""\n\nRespond with a theme based on the product_sku, programming language and/or spend. Be specific about a theme with a maxium of three words, pick something specifc to the data you are reviewing based on the data of product_sku "sms", "voice", "flex", "segment", "sendgrid", "sms", "sms"] and or programming language:',
                temperature=0.5,
                max_tokens=64,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            theme = response["choices"][0]["text"].replace("\n", "")
            themes.append(theme)
            embedding_cache[reviews] = theme

    st.write(themes)
    # save the embedding cache to a pickle file
    with open(embedding_cache_file, "wb") as cache_file:
        pickle.dump(embedding_cache, cache_file)

    # visualize the clusters using streamlit

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        customer_embeddings[:, 0], customer_embeddings[:, 1], c=labels, cmap='viridis')

    # add colorbar to the plot
    cbar = plt.colorbar(scatter)

    # customize plot labels and title
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Customer Clusters")

    # label the clusters by writing out the colors and themes
    # Create a legend with the cluster names and colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label=f"Cluster {cluster}", markerfacecolor=color, markersize=10)
        for cluster, color in zip(clusters, scatter.get_cmap()(np.unique(labels) / (n_clusters - 1)))
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Annotate clusters with themes and display cluster names
    for i, cluster in enumerate(clusters):
        ax.annotate(themes[i], (customer_embeddings[i, 0],
                    customer_embeddings[i, 1]))

        # Get the color of the cluster
        color = scatter.get_cmap()(labels[i] / (n_clusters - 1))

        # Display the cluster name with color using st.write
        st.write(
            f"Cluster {cluster}: <span style='color:{color};'>{themes[i]}</span>", unsafe_allow_html=True)

    # display the plot using streamlit
    st.pyplot(fig)

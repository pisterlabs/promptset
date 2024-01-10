import boto3
import os
import re
import openai
import backoff
import hdbscan

import numpy as np
import pandas as pd

from openai.error import RateLimitError

def gen_file_name(cat, pub_years):
    """
    Generates a filename based on a given category and list of publication years.

    Parameters:
    cat (str): Subject category.
    pub_years (list): List of publication years.

    Returns:
    str: A filename in the format "category_year1_year2...yearN.parquet".
         Spaces in the category are replaced with underscores.
         Years are sorted in ascending order and separated by underscores.
    """
    return f"{cat.replace(' ', '_')}_{'_'.join([str(x) for x in sorted(pub_years)])}.parquet"

def search_s3(file_name):
    """
    Checks if a file exists in a specific S3 bucket location.

    Parameters:
    file_name (str): Name of the file to be searched in the S3 bucket.

    Returns:
    bool: True if the file exists, False otherwise.
    
    Notes:
    Uses AWS Boto3 to establish a session with given AWS credentials.
    The file is searched in the 'rootbucket' bucket under 'topic_clustering/test_folder/umaps/'.
    """
    session = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    client = session.client('s3')
    try:
        client.head_object(Bucket='rootbucket', Key=f'topic_clustering/test_folder/umaps/{file_name}')
        return True
    except:
        return False 

def write_stub_s3(file_name, user):
    """
    Writes or updates a stub file in the S3 bucket with the user's name.

    Parameters:
    file_name (str): Name of the file corresponding to the stub to be written.
    user (str): User's name to be added to the stub file.

    Returns:
    bool: True if the stub file is updated or written successfully, False if a new file is created.

    Notes:
    Checks and updates a .txt file replacing its .parquet extension in the same directory.
    If the user's name is not present in the existing file, it is appended.
    """
    session = boto3.Session(aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    try:
        client = session.client('s3')
        data = client.get_object(Bucket='rootbucket', Key=f'topic_clustering/test_folder/stubs/{file_name.replace("parquet", "txt")}')
        contents = data["Body"].read().decode('utf-8').split(",")
        if user not in contents:
            contents.append(user)
        contents = ",".join(contents)
        s3 = session.resource('s3')
        object = s3.Object("rootbucket", f"topic_clustering/test_folder/stubs/{file_name.replace('parquet', 'txt')}")
        object.put(Body=contents)
        return True
    except:
        s3 = session.resource('s3')
        object = s3.Object("rootbucket", f"topic_clustering/test_folder/stubs/{file_name.replace('parquet', 'txt')}")
        object.put(Body=user)
        return False
    

def get_subject(file_name):
    subject = [x for x in re.findall(r'\D+', file_name) if x not in ['.txt', '_']]
    return [x.replace("_", " ").strip() for x in subject][0]

def get_pub_years(file_name):
    years = re.findall(r"\d+", file_name)
    return [int(x) for x in years]

openai.api_key = os.getenv("OPENAI_TOPIC_CLUSTERING")

@backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
def generate_label(article_titles):
    """
    Generates a label for a set of article titles using OpenAI's GPT model.

    Parameters:
    article_titles (list): A list of article titles.

    Returns:
    str: A label generated based on the combined research topics of the articles.

    Notes:
    Utilizes OpenAI's ChatCompletion API.
    The function is decorated with backoff to handle rate limit errors.
    """
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=1,
        messages=[
            {
                "role": "user",
                "content": f"What label would you give the combined research described by these paper titles by the same author? Return the label only with no other text: {article_titles}",
            }
        ],
    )
    label = completion.choices[0].message.content.replace("\n", "")
    return label

def create_gpt_label_dataframe(exemplars):
    """
    Creates a DataFrame with GPT-generated labels for clusters of articles.

    Parameters:
    exemplars (DataFrame): DataFrame with article titles and their cluster labels.

    Returns:
    DataFrame: A DataFrame containing cluster labels and corresponding GPT-generated labels.

    Notes:
    For each unique cluster, a sample of 20 article titles is chosen to generate a label.
    The resulting DataFrame includes a 'cluster_label' and a 'gpt_label' for each cluster.
    """
    gpt_labels = pd.DataFrame()
    for cluster in exemplars.cluster_label.unique().tolist():
        cluster_exemplars = exemplars["article_title"][exemplars["cluster_label"]==cluster].sample(20).tolist()
        label = generate_label(cluster_exemplars)
        gpt_labels = pd.concat([gpt_labels, pd.DataFrame({"cluster_label":cluster, "gpt_label":label}, index=[0])])
    return gpt_labels

def initalise_clusterer(df, min_cluster_size, min_samples, cluster_selection_method, cluster_selection_epsilon, metric):
    """
    Initializes an HDBSCAN clusterer with specified parameters.

    Parameters:
    df (DataFrame): DataFrame containing the UMAP coordinates.
    min_cluster_size (int): Minimum cluster size.
    min_samples (int): Minimum samples in a neighborhood for a core point.
    cluster_selection_method (str): Method used for cluster selection.
    cluster_selection_epsilon (float): Epsilon value for cluster selection.
    metric (str): Distance metric for clustering.

    Returns:
    HDBSCAN: An HDBSCAN clustering object fitted to the data.

    Notes:
    The function takes UMAP coordinates from the input DataFrame for clustering.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
    ).fit(df[["coord_x", "coord_y"]])
    return clusterer

def get_cluster_labels(df, params):
    """
    Assigns HDBSCAN cluster labels to each document in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing UMAP coordinates.
    params (dict): Parameters for HDBSCAN clustering, including minimum cluster size, 
                   minimum samples, cluster selection method, epsilon, and metric.

    Returns:
    tuple: A tuple containing (1) the updated DataFrame with cluster labels and exemplar flags, 
           (2) the percentage of points clustered, and (3) the total number of clusters.

    Notes:
    - The function initializes an HDBSCAN clusterer using the given parameters.
    - Cluster labels are added to the DataFrame. '-1' indicates points not assigned to any cluster.
    - Exemplar points for each cluster are identified and merged into the DataFrame.
    - The function calculates and returns the percentage of points clustered and the total number of clusters.
    """
    clusterer = initalise_clusterer(df, params["min_cluster_size"], params["min_samples"], params["cluster_selection_method"], params["cluster_selection_epsilon"], params["metric"])
    df["cluster_label"] = clusterer.labels_
    pct_clustered = 1 - np.count_nonzero(clusterer.labels_ == -1) / len(clusterer.labels_)
    number_clusters = clusterer.labels_.max() + 1

    exemplars = pd.DataFrame()
    for index, cluster in enumerate(clusterer.exemplars_):
        cluster_exemplars = pd.DataFrame(cluster, columns=["coord_x", "coord_y"])
        cluster_exemplars["exemplar"] = True    
        exemplars = pd.concat([exemplars, cluster_exemplars], axis=0)
        
    df = df.merge(exemplars, on=["coord_x", "coord_y"], how="left")
    df["exemplar"] = df["exemplar"].fillna(False)

    return df, pct_clustered, number_clusters

def group_authors(articles, authors):
    """
    Groups authors based on GPT-generated labels, citations, and other metadata.

    Parameters:
    articles (DataFrame): DataFrame containing article information like DOI, GPT label, and citations.
    authors (DataFrame): DataFrame containing author information.

    Returns:
    DataFrame: A DataFrame of authors grouped by GPT labels and other metadata, with calculated metrics.

    Notes:
    - Merges author data with article data based on DOIs.
    - Groups authors by GPT label, author name, research organization, country, and region.
    - Aggregates data to calculate total publications, citations, and lists of source titles and publishers per group.
    - Calculates the average citations per article for each author group.
    - Cleans special characters from specific columns and converts certain lists to string representations.
    """
    df_authors = authors.merge(articles[["doi", "gpt_label", "citations"]], on="doi", how="left").drop_duplicates()
    df_authors = df_authors[df_authors["gpt_label"].notna()]

    def collect_group(group):       
        return list(dict.fromkeys(group))

    authors_grouped = (
        df_authors.groupby(
            [
                "gpt_label",
                "author_full_name",
                "research_org",
                "prid_country",
                "prid_region",
            ]
        )
        .agg(
            sum_published=pd.NamedAgg(
                column="author_full_name", aggfunc="count"
            ),
            sum_citations=pd.NamedAgg(column="citations", aggfunc="sum"),
            full_source_title_list=pd.NamedAgg(column="full_source_title", aggfunc=collect_group),
            publisher_group_list=pd.NamedAgg(column="publisher_group", aggfunc=collect_group)
        )
        .reset_index()
        .sort_values(
            by=["gpt_label", "sum_published", "sum_citations"],
            ascending=[True, False, False],
        )
    ).sort_values("sum_citations", ascending=False)
    
    for col in ["author_full_name","research_org","prid_country","prid_region"]:
        authors_grouped[col] = authors_grouped[col].apply(lambda x: re.sub(r'\W+', '', x))

    authors_grouped["avg_cites_per_article"] = (authors_grouped["sum_citations"] / authors_grouped["sum_published"]).astype(int) 

    authors_grouped = authors_grouped.reset_index(drop=False, names="index")

    authors_grouped["full_source_title_list"] = authors_grouped["full_source_title_list"].apply(lambda x: str(x))
    authors_grouped["publisher_group_list"] = authors_grouped["publisher_group_list"].apply(lambda x: str(x))

    return authors_grouped    
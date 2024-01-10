import os
import openai
from tqdm import tqdm
import pandas as pd

from azure.identity import DefaultAzureCredential
from pprint import pprint

credential = DefaultAzureCredential()


token = credential.get_token("https://cognitiveservices.azure.com/.default",
                             tenant_id='ef12d42b-60e3-4161-87ee-98ebcc73eae8')

pprint(dir(token))

openai.api_type = "azure"
openai.api_key = os.getenv("DEV_OAIAPI")
openai.api_base = "https://dev-oaiapi.openai.azure.com/"
openai.api_version = "2023-05-15"  # subject to change


# Move get_embedding to the global scope
def get_embedding(text_to_embed):
    embedding = openai.Embedding.create(
        input=text_to_embed, deployment_id="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return embedding


emb = get_embedding(text_to_embed='Hi there!')

def get_ada_embeddings():
    # Get the reviews
    reviews = pd.read_parquet('data/google_maps_reviews.parquet', engine='pyarrow')

    # Get non-null reviews
    reviews = reviews[reviews['snippet'].notnull()]

    # Get Quartile 1 character length
    reviews['snippet'].str.len().quantile(0.25)
    reviews['snippet'].str.len().quantile(0.50)
    reviews['snippet'].str.len().quantile(0.75)


    # Get OPENAI KEY
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    # Initialize tqdm with pandas
    tqdm.pandas()

    # Apply the function and show a progress bar
    reviews["embedding"] = reviews["snippet"].astype(str).progress_apply(get_embedding)

    # Write the reviews to a parquet file
    reviews.to_parquet('data/google_maps_reviews_with_embeddings.parquet', engine='pyarrow')


if __name__ == '__main__':
    get_ada_embeddings()


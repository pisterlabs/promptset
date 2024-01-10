"""Build a pinecone index with the NHS Start for Life data."""

import os
import uuid

import openai
import pandas as pd

from dotenv import load_dotenv

from genai.eyfs import get_embedding
from genai.vector_index import PineconeIndex


load_dotenv()


openai.api_key = os.environ["OPENAI_API_KEY"]
INDEX_NAME = "eyfs-index"
ENCODER_NAME = "text-embedding-ada-002"

if "__main__" == __name__:
    df = pd.read_csv(os.environ["PATH_TO_NHS_DATA"])
    df = df.groupby("URL").apply(lambda group: "\n\n".join(group["header"] + "\n" + group["content"]))
    df = df.reset_index()
    df.columns = ["URL", "content"]

    # Format the data to what pinecone needs and generate a temp uuid
    docs = []
    for tup in df.itertuples():
        doc = tuple(
            (
                str(uuid.uuid4()),
                get_embedding(tup.content),
                {
                    "source": "nhs_full_page",
                    "text": tup.content,
                    "url": "".join(["https://www.nhs.uk", tup.URL]),
                },
            )
        )
        docs.append(doc)

    # Build the index
    conn = PineconeIndex(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_REGION"])

    conn.build_and_upsert(
        index_name=INDEX_NAME,
        dimension=1536,
        metric="euclidean",
        docs=docs,
        metadata_config={"indexed": ["areas_of_learning", "source", "type_", "age_group"]},
        batch_size=40,
    )

"""Build a pinecone index with the Development Matters learning goals and examples."""

import os
import uuid

import openai

from dotenv import load_dotenv

from genai.eyfs import get_embedding
from genai.utils import read_json
from genai.vector_index import PineconeIndex


load_dotenv()


openai.api_key = os.environ["OPENAI_API_KEY"]
PATH_TO_DM = "src/genai/dm/dm.json"
INDEX_NAME = "eyfs-index"
ENCODER_NAME = "text-embedding-ada-002"

if "__main__" == __name__:
    data = read_json(PATH_TO_DM)
    # Temp hack to exclude the template
    data = [d for d in data if d["area_of_learning"] != ""]

    # Format the data to what pinecone needs and generate a temp uuid
    docs = []
    for elem in data:
        aol = elem["area_of_learning"]
        d = elem["age_group"]
        for age, age_dict in d.items():
            for k, items in age_dict.items():
                for item in items:
                    doc = tuple(
                        (
                            str(uuid.uuid4()),
                            get_embedding(item),
                            {
                                "age_group": age,
                                "type_": k,
                                "source": "dm",
                                "text": item,
                                "areas_of_learning": aol,
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

import json
import os

import openai
import pinecone
from dotenv import load_dotenv

load_dotenv()
index_name = "tiro-papers-1"
environment = "us-east-1-aws"
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
open_api_key = os.environ.get("OPENAI_API_KEY")


def upload_summary():
    # setup pinecone
    pinecone.init(api_key=pinecone_api_key, environment=environment)
    index = pinecone.Index(index_name=index_name)

    # load summary file
    f = open("/Users/shmoon/Desktop/projects/t-backend/t_prompt/paper_values.json", "r")
    paper_values = json.load(f)
    vectors = []

    for paper_id, paper_value in enumerate(paper_values):
        title = paper_value["title"]
        summary = paper_value["summary"]
        values = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=summary,
            api_key=open_api_key,
        )
        vectors.append(
            {
                "id": "paper-" + str(paper_id),
                "values": values.data[0].embedding,
                "metadata": {"title": title},
            }
        )

    index.upsert(vectors=vectors)
    print("done")


if __name__ == "__main__":
    upload_summary()

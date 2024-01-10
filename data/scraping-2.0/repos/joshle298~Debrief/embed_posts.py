from langchain.vectorstores import AtlasDB
from langchain.embeddings.openai import OpenAIEmbeddings
import nomic
import os
import json
import uuid

# load API keys from .env file
import dotenv
dotenv.load_dotenv()

# login to Nomic Atlas
nomic.login(os.getenv("ATLAS_TEST_API_KEY"))

# create a vectorstore
vectorstore = nomic.AtlasProject(
    name="Debrief",
    reset_project_if_exists=True,
    is_public=True,
    unique_id_field="id_field",
    modality="text",
)

def datafile_to_embedding_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    print(filename)

    if isinstance(data, list):
        items = data
        feed_title = ""
        feed_link = ""
    else:
        items = data.get("items", [])
        feed_title = data["title"] if "title" in data else ""
        feed_link = data["link"] if "link" in data else ""

    metadata = [
        {
            "id_field": str(uuid.uuid4()),
            "embed_text": " - ".join([x["title"], x["description"]])
            if "description" in x
            else x["title"],
            "title": x["title"],
            "description": x["description"] if "description" in x else "",
            "link": x["link"] if "link" in x and x["link"] is not None else "",
            "pubDate": x["pubDate"] if "pubDate" in x else "",
            "feed_title": feed_title,
            "feed_link": feed_link,
        }
        for x in items
    ]
    return metadata

# cnn_metadata = datafile_to_embedding_data("data_sources/cnn_rss_data.json")
# techcrunch_metadata = datafile_to_embedding_data(
#     "data_sources/techcrunch_rss_data.json"
# )

# for each file in data_sources load the json and add it to the vectorstore
for filename in os.listdir("data_sources"):
    if filename.endswith(".json"):
        metadata = datafile_to_embedding_data(os.path.join("data_sources", filename))
        vectorstore.add_text(data=metadata)

# add the following files in the same manner from the scrapers directory
files = [
    "scrapers/hn.json",
    "scrapers/reddit_posts.json",
    "scrapers/tweets.json",
]

# create a vectorstore index
for filename in files:
    metadata = datafile_to_embedding_data(filename)
    vectorstore.add_text(data=metadata)

# create text embeddings
vectorstore.create_index(
    name="v1.1",
    indexed_field="embed_text",
    build_topic_model=True,
    topic_label_field="embed_text",
    colorable_fields=["feed_title", "id_field"],
)

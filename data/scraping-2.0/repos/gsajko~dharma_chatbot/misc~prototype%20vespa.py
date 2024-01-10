# %%
import hashlib
import os
from typing import Iterable

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vespa.deployment import VespaDocker
from vespa.io import VespaQueryResponse, VespaResponse
from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    RankProfile,
    Schema,
)

# %%
# Define and deploy a Vespa application package using PyVespa.

markdown_schema = Schema(
    name="markdown",
    document=Document(
        fields=[
            Field(name="id", type="string", indexing=["summary", "index"]),
            Field(
                name="chunks",
                type="array<string>",
                indexing=["summary", "index"],
                index="enable-bm25",
            ),
        ],
    ),
)
# add ranking profile
# TODO
basic_bm25 = RankProfile(name="bm25", first_phase="bm25(chunks)")

markdown_schema.add_rank_profile(basic_bm25)
# Create the application package
vespa_app_name = "rag"
app_package = ApplicationPackage(name=vespa_app_name, schema=[markdown_schema])

# deploy
vespa_docker = VespaDocker()
vespa_app = vespa_docker.deploy(application_package=app_package)
print("Deployment successful ✨")


# %%
# Utilize LangChain to parse markdown files.
# get all markdown files from folder

folder_path = "data/md_marker"
md_list = []
for file in os.listdir(folder_path):
    if file.endswith(".md"):
        md_list.append(file)

# open all files, transform them into Langchain documents
# chunk them
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1024,
    chunk_overlap=24,
    length_function=len,
    add_start_index=True,
)
docs = []
for path in md_list:
    loader = TextLoader(f"{folder_path}/{path}")
    data = loader.load_and_split(text_splitter=text_splitter)
    text_chunks = [chunk.page_content for chunk in data]
    vespa_id = path
    hash_value = hashlib.sha1(vespa_id.encode()).hexdigest()
    fields = {
        "id": hash_value,
        "chunks": text_chunks,
    }
    print(len(text_chunks))
    docs.append(fields)
print("Documents created ✨")
# %%
# Feed the docs to the running Vespa instance.


def vespa_feed(user: str) -> Iterable[dict]:
    for doc in docs:
        yield {"fields": doc, "id": doc["id"], "groupname": user}


def callback(response: VespaResponse, id: str):
    if not response.is_successful():
        print(
            f"Document {id} failed to feed with status code {response.status_code},"
            f"url={response.url} response={response.json}"
        )
    if response.is_successful():
        print(f"Document {id} successfully fed")
        print(response.json["pathId"])
        # print(response.json)


namespace = "personal"
group_id = "gsajko"
schema_name = "markdown"

vespa_app.feed_iterable(
    schema=schema_name,
    iter=vespa_feed(user=group_id),
    namespace=namespace,
    callback=callback,
)
# %%
query_terms = (
    "Jewel of the Powerful Nāgārjuna's Intention AND"
    " samādhi practices AND Cultivation Of Insight"
)
response = vespa_app.query(
    yql="select * from markdown where userQuery();",
    rank_profile="bm25",
    query=query_terms,
)

# %%

response: VespaQueryResponse = vespa_app.query(
    yql="select * from markdown where true",
    groupname=group_id,
    query="what is attention?",
    rank_profile="nativeRank",
)
assert response.is_successful()
len(response.hits)

# %%
for i, hit in enumerate(response.hits, start=1):
    print(f"Document {i}: ID = {hit['id']}, Relevance = {hit['relevance']}")
# %%
response = vespa_app.query(
    yql="select * from markdown where true;", hits=5  # Adjust as needed
)

if response.is_successful():
    for i, hit in enumerate(response.hits, start=1):
        doc_id = hit["id"]
        chunks = hit.get("fields", {}).get("chunks", "No chunks field")
        print(f"Document {i}: ID = {doc_id}, # Chunks = {len(chunks)}")
else:
    print(
        f"Query failed with status code {response.status_code}."
        f" Reason: {response.json()}"
    )

# %%
response = vespa_app.query(
    yql="select chunks from markdown where true;", hits=5  # Adjust as needed
)

if response.is_successful():
    for i, hit in enumerate(response.hits, start=1):
        doc_id = hit["id"]
        chunks = hit.get("fields", {}).get("chunks", "No chunks field")
        print(f"Document {i}: ID = {doc_id}, Chunks = {chunks}")
else:
    print(
        f"Query failed with status code {response.status_code}."
        f"Reason: {response.json()}"
    )

# %%
query_terms = (
    "Jewel of the Powerful Nāgārjuna's Intention AND "
    "samādhi practices AND Cultivation Of Insight"
)
response = vespa_app.query(
    yql="select * from markdown where userQuery();",
    rank_profile="bm25",
    query=query_terms,
)


# %%

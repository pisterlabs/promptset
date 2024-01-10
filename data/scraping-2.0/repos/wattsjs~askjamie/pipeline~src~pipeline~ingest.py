import json
import logging
import os
import sys
import time
from uuid import uuid4

from langchain.document_loaders import (
    PlaywrightURLLoader,
    OnlinePDFLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import Document
from openai.error import APIError
from pydantic.parse import load_file
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchText,
    PointStruct,
)

from pipeline.tokens import text_splitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or "QDRANT_API_KEY"
QDRANT_URL = os.getenv("QDRANT_URL") or "QDRANT_URL"
model_name = "text-embedding-ada-002"
collection_name = "askcisco.com"


def main():
    qdrant_client = QdrantClient(
        url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True
    )
    embed = OpenAIEmbeddings(client=None, model=model_name, show_progress_bar=True)

    create_collection(qdrant_client)

    # check if --force flag is passed
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        force = True

    ingest(qdrant_client, embed, "docs", force=force)
    ingest(qdrant_client, embed, "pdfs", force=force)
    ingest(qdrant_client, embed, "urls", force=force)


def ingest(client: QdrantClient, embed: OpenAIEmbeddings, type: str, force=False):
    datas = get_queued_data(type)

    for data in datas:
        # check if "force" or "update" is a key in the data
        document_force = False
        if data.get("force") or data.get("update"):
            document_force = True

        if not force and not document_force:
            results, _ = client.scroll(
                collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.source",
                            # match on text because there may be a # in the url
                            match=MatchText(text=data["source"].replace(".html", "")),
                        )
                    ]
                ),
            )
            if len(results):
                logging.info(f"⤵️ skipping {data['source']} - already exists")
                continue
        else:
            delete_result = client.delete(
                collection_name,
                points_selector=FilterSelector(
                    filter=models.Filter(
                        must=[
                            FieldCondition(
                                key="metadata.source",
                                # match on text because there may be a # in the url
                                match=MatchText(
                                    text=data["source"].replace(".html", "")
                                ),
                            )
                        ]
                    )
                ),
            )
            logging.info(f"🗑 deleted {delete_result}")
            logging.info(f"‼️ force adding {data['source']}")

            # if the document was forced, remove the force key
            if "force" in data:
                del data["force"]
            if "update" in data:
                del data["update"]

            update_queued_data(type, datas)

        if type == "urls":
            docs = get_documents_from_queued_urls([data])
        elif type == "docs":
            docs = get_documents_from_queued_docs([data])
        elif type == "pdfs":
            docs = get_docs_from_queued_pdfs([data])
        else:
            raise Exception("unknown type")

        new_embeddings = []

        docs_content = [d.page_content for d in docs]
        batch = 100
        for i in range(0, len(docs_content), batch):
            try:
                logging.info(f"adding documents {i} to {i+batch}")
                new_embeddings_batch = embed.embed_documents(
                    docs_content[i : i + batch]
                )
                new_embeddings.extend(new_embeddings_batch)
                time.sleep(0.1)
            except APIError:
                logging.error(
                    "⚠️ openai api error - waiting 60 seconds and retrying..."
                )
                time.sleep(30)
                new_embeddings_batch = embed.embed_documents(
                    docs_content[i : i + batch]
                )
                new_embeddings.extend(new_embeddings_batch)

        if not new_embeddings:
            logging.info(f"⤵️ skipping {data['source']} - no embeddings")
            continue

        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=docs[idx].dict(),
                )
                for idx, vector in enumerate(new_embeddings)
            ],
        )

        logging.info(f"🧠 added {len(new_embeddings)} new {type} embeddings")


def create_collection(client: QdrantClient, dimension: int = 1536):
    collections = client.get_collections()
    collection_names = [c.name for c in collections.collections]

    # only create collection if it doesn't exist
    if collection_name not in collection_names:
        logging.info(f"creating collection {collection_name}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )


def get_documents_from_queued_docs(docs: list[dict] | None = None):
    """Get all queued scraped docs if they exist"""
    if not docs:
        docs = get_queued_data("docs")
    if not docs:
        return []

    logging.info(f"processing {len(docs)} docs")

    parsed_docs = []

    for doc in docs:
        if "source" not in doc:
            logging.warning(f"no source found for {doc}")
            continue

        logging.info(f"📄 processing {doc['source']}")

        try:
            # load the document as json
            slug = doc["slug"]
            versions = doc.get("versions", [])
            if "version" in doc:
                versions.append(doc["version"])

            if not versions:
                file_name = f"data/queue/docs/{slug}.json"
            else:
                file_name = f"data/queue/docs/{slug}-{versions[0]}.json"

            sections = load_file(file_name)
        except Exception as e:
            logging.error(f"failed to parse {doc['source']}: {e}")
            continue

        products = doc.get("products", [])
        if "product" in doc:
            products.append(doc["product"])

        if "firepower" in products or "firewall" in products:
            product = "Secure Firewall"
        elif "ise" in products:
            product = "Identity Services Engine"
        elif "umbrella" in products:
            product = "Cisco Umbrella"
        else:
            product = products[0].title()

        for section in sections:
            meta = doc

            meta.update(
                {
                    "slug": doc["slug"],
                    "title": section["title"],
                    "subtitle": section["subtitle"]
                    if "subtitle" in section
                    else section["header"]
                    if "header" in section
                    else None,
                    "source": section["url"],
                }
            )

            if versions:
                meta["versions"] = versions

            sec = Document(
                page_content=section["content"],
                metadata=meta,
            )
            parsed_docs.extend(text_splitter.split_documents([sec]))

    return parsed_docs


def get_documents_from_queued_urls(urls: list[dict] | None = None) -> list[Document]:
    """Get all urls from the urls.json file in data/queue"""
    # open the urls.json file and read its contents
    if not urls:
        urls = get_queued_data("urls")
    if not urls:
        return []

    logging.info(f"processing url {urls[0]['source']} ...")

    loader = PlaywrightURLLoader(
        urls=[u["source"] for u in urls], remove_selectors=["header", "footer"]
    )
    data = loader.load()

    split_docs = text_splitter.split_documents(data)

    # load any metadata from the urls.json file into the document
    for doc in split_docs:
        for url in urls:
            if doc.metadata["source"].strip(".") == url["source"].strip("."):
                doc.metadata.update(url)
                if "version" in doc.metadata:
                    doc.metadata["versions"] = [doc.metadata["version"]]

    return split_docs


def get_docs_from_queued_pdfs(pdfs: list[dict] | None = None) -> list[Document]:
    # open the urls.json file and read its contents
    if not pdfs:
        pdfs = get_queued_data("pdfs")
    if not pdfs:
        return []

    logging.info(f"processing pdf {pdfs[0]['source']} ...")

    docs = []

    for pdf in pdfs:
        try:
            loader = OnlinePDFLoader(pdf["source"])
            doc = loader.load()

            # replace some boilerplate
            if doc:
                doc[0].page_content = (
                    doc[0]
                    .page_content.replace("Cisco Public", "")
                    .replace("All Rights Reserved", "")
                )

            pages = text_splitter.split_documents(doc)
        except Exception as e:
            logging.error(f"failed to parse {pdf['source']}: {e}")
            continue
        # load any metadata from the urls.json file into the document
        for doc in pages:
            doc.metadata.update(pdf)
        docs.extend(pages)

    return docs


def get_queued_data(type: str):
    data: list[dict] = []
    try:
        with open(f"pipeline/data/queue/{type}.json", "r") as f:
            d = f.read()
            if not d:
                logging.info(f"no {type}s to process")
                return data
            data = json.loads(d)
            data = [u for u in data if "source" in u]
    except FileNotFoundError:
        with open(f"data/queue/{type}.json", "r") as f:
            d = f.read()
            if not d:
                logging.info(f"no {type}s to process")
                return data
            data = json.loads(d)
            data = [u for u in data if "source" in u]

    return data


def update_queued_data(type: str, data: list[dict]):
    # override the existing file with the new data
    try:
        with open(f"data/queue/{type}.json", "w") as f:
            f.write(json.dumps(data))
    except FileNotFoundError:
        with open(f"pipeline/data/queue/{type}.json", "w") as f:
            f.write(json.dumps(data))

    logging.info(f"✅ updated {type}.json")


if __name__ == "__main__":
    main()

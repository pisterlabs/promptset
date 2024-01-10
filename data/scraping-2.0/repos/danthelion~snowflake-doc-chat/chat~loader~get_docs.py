import os
from langchain.document_loaders.sitemap import SitemapLoader
from utils.document_serde import save_docs_to_jsonl
from utils.constants import RAW_DATA_DIR


def get_docs():
    sitemap_loader = SitemapLoader(
        web_path="https://docs.snowflake.com/sitemap.xml",
    )

    sitemap_loader.requests_per_second = 4

    serialized_docs_path = f"{RAW_DATA_DIR}/data.jsonl"

    if os.path.exists(serialized_docs_path):
        print(f"Skipping {serialized_docs_path} as it already exists")
        return

    docs = sitemap_loader.load()

    save_docs_to_jsonl(docs, serialized_docs_path)
    print(f"Saved {len(docs)} documents to {serialized_docs_path}")

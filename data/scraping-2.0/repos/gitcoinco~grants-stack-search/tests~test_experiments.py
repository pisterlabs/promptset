from langchain.document_loaders import JSONLoader
from rake_nltk import Rake
import pytest
import nltk
from pprint import pprint
from src.data import Data
from src.util import ApplicationFileLocator


# NOTE: these are not real tests, we're using pytest as a testbed.


@pytest.mark.skip
def test_ingest_and_persist(tmp_path):
    Data.ingest_and_persist(
        application_files_locators=[
            ApplicationFileLocator(
                round_id="0x98720dD1925d34a2453ebC1F91C9d48E7e89ec29", chain_id=424
            ),
        ],
        storage_dir=tmp_path,
        indexer_base_url="https://indexer-staging.fly.dev",
        first_run=True,
    )


@pytest.mark.skip
def test_experiment_keyword_extraction_with_rake():
    nltk.download("stopwords")
    nltk.download("punkt")

    def get_metadata(record: dict, metadata: dict) -> dict:
        metadata["name"] = record.get("title")
        metadata["website_url"] = record.get("website")
        return metadata

    loader = JSONLoader(
        file_path=".var/1/projects.shortened.json",
        jq_schema=".[].metadata | { title, website, description }",
        content_key="description",
        metadata_func=get_metadata,
        text_content=False,
    )

    docs = loader.load()
    docs_with_title = [d.metadata["name"] + "\n\n" + d.page_content for d in docs]
    text = "\n\n".join(docs_with_title)
    r = Rake()
    r.extract_keywords_from_text(text)
    pprint(r.get_ranked_phrases_with_scores()[0:30])

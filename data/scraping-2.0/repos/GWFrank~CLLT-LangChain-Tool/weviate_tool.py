import dataclasses
from dataclasses import dataclass
import os
import weaviate
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ContentItem:
    media: str  # media source of the post or comment
    content_type: str  # post or comment
    author: str  # author of the post or comment
    post_id: str  # id of the post
    year: str  # year of the post
    board: str  # board of the post
    title: str  # title of the post
    text: str  # text of the post or comment
    rating: str  # rating of the comment
    order: int  # 0 for post, 1, 2, 3, ... for comments
    chunk: int  # if text too long, split into chunks
    total_chunks: int  # total number of chunks


client = weaviate.Client(
    url=os.environ["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthApiKey(
        api_key=os.environ["WEAVIATE_ADMIN_PASS"]),
    timeout_config=(5, 30),  # type: ignore
    additional_headers={'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]}
)

attributes = [field.name for field in dataclasses.fields(ContentItem)]


def retrieve_docs(keyword, count=5):
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        k=count,
        # weighting for each search algorithm (alpha = 0 (sparse, BM25), alpha = 1 (dense), alpha = 0.5 (equal weight for sparse and dense))
        alpha=0.5,
        index_name="ContentItem",
        text_key="text",
        attributes=attributes,
    )
    return retriever.get_relevant_documents(keyword)

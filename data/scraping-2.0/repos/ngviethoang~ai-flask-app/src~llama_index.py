import os
import re
import traceback
from typing import List, Literal

import requests
from newspaper import Article
from youtube_transcript_api import YouTubeTranscriptApi

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import (
    Document,
    GPTSimpleVectorIndex,
    LLMPredictor,
    NotionPageReader,
    PromptHelper,
    ServiceContext,
)
from llama_index.readers.schema.base import Document
from .utils.notion import query_database, create_page


NOTION_API_KEY = os.getenv("NOTION_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

notion_database_id = os.getenv("URL_NOTION_DATABASE_ID")


def get_notion_item(url: str):
    filter_db = {"and": [{"property": "URL", "url": {"equals": url}}]}

    result = query_database(notion_database_id, filter_db)

    return None if not result["results"] else result["results"][0]


def create_notion_item(article: Article):
    def text_to_blocks():
        lines = article.text.split("\n")
        blocks = []

        for line in lines:
            if line:
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": line,
                                    },
                                }
                            ]
                        },
                    }
                )

        return blocks

    result = create_page(
        parent={"database_id": notion_database_id},
        properties={
            "Title": {"title": [{"text": {"content": article.title}}]},
            "URL": {"url": article.url},
        },
        children=text_to_blocks(),
    )
    return result


def get_index(
    documents: List[Document],
    model_name: str,
):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2056
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit=chunk_size_limit,
    )

    # define LLM
    llm_predictor = LLMPredictor(
        # llm=OpenAI(
        #     temperature=0,
        #     model_name=model_name or "gpt-3.5-turbo",
        #     max_tokens=num_outputs,
        #     openai_api_key=OPENAI_API_KEY,
        # ),
        llm=ChatOpenAI(
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
            max_tokens=num_outputs,
        ),
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    return index


def get_notion_documents(
    page_ids: List[str],
):
    reader = NotionPageReader(integration_token=NOTION_API_KEY)
    documents = reader.load_data(page_ids=page_ids)

    return documents


def handle_url(
    url: str,
    prompt: str,
    prompt_type: Literal["summarize", "qa"],
    model_name: str,
):
    try:
        video_id = get_youtube_video_id(url)
        if video_id:
            documents = get_documents(ids=[video_id], languages=["en", "vi"])

        else:
            # normal URL
            item = get_notion_item(url)

            if not item:
                article = Article(url)
                article.download()
                article.parse()
                # article.nlp()

                item = create_notion_item(article)

            documents = get_notion_documents([item["id"]])

        index = get_index(documents, model_name)

        response_mode = "tree_summarize" if prompt_type == "summarize" else "default"
        response = index.query(prompt + "\n", response_mode=response_mode)
        return response.response.strip()
    except Exception as e:
        traceback.print_exc()
        return None if e.args else e.args[0]


def get_youtube_video_id(url: str):
    # Regular expression to match YouTube video URLs
    regex = r"(?:https?:\/\/)?(?:[0-9A-Z-]+\.)?(?:youtube|youtu|youtube-nocookie)\.(?:com|be)\/(?:watch\?v=|watch\?.+&v=|embed\/|v\/|.+\?v=)?([^&=\n%\?]{11})"

    match = re.match(regex, url, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2) or match.group(3)

    # If the URL is not a YouTube video, return None
    return None


def get_documents(ids: List[str], languages: List[str]):
    results = []
    for id in ids:
        srt = YouTubeTranscriptApi.get_transcript(id, languages=languages)
        transcript = ""
        for chunk in srt:
            transcript = transcript + chunk["text"] + "\n"
        results.append(Document(transcript))
    return results

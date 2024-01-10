from typing import List

from fastapi import APIRouter
from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from api.schemas import youtube as youtube_schemas
from api.settings import youtube as youtube_settings

settings = youtube_settings.YouTubeSettings()
router = APIRouter()

PROMPT_TEMPLATE = """Write a concise Japanese summary of the following transcript of Youtube Video.
============

{text}

============

日本語で 500 文字以内で要約した結果は以下の通りです。
"""


def create_llm():
    return ChatOpenAI(
        model_name=settings.openai_model_name,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        temperature=0,
        model_kwargs={
            "deployment_id": settings.openai_deployment_id,
            "api_type": settings.openai_api_type,
            "api_version": settings.openai_api_version,
        },
    )


def get_documents(url: str) -> str:
    docs = YoutubeLoader.from_youtube_url(
        url, add_video_info=True, language=["en", "ja"]
    ).load()
    if len(docs) == 0:
        raise ValueError("No documents found.")
    return docs[0].page_content


def split_docs(text: str) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", "", "。", "!", "！", "?", "？"],
        chunk_size=15000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )

    texts = text_splitter.split_text(text)
    print(f"splitted to {len(texts)} chunks")
    return [Document(page_content=t) for t in texts]


def summarize(llm: ChatOpenAI, docs: [Document]):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])
    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=prompt,
            combine_prompt=prompt,
            collapse_prompt=prompt,
            verbose=False,
        )

        output = chain(
            inputs=docs,
            return_only_outputs=True,
        )["output_text"]
    return output, cb.total_cost


@router.post("/youtube/summarize", response_model=youtube_schemas.SummarizeResponse)
async def chat(body: youtube_schemas.SummarizeRequest):
    llm = create_llm()
    transcript = get_documents(body.url)
    docs = split_docs(transcript)
    summary, cost = summarize(llm, docs)

    return youtube_schemas.SummarizeResponse(
        content=summary,
        cost=cost,
    )

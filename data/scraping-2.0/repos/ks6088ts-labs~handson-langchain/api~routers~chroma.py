import uuid
from typing import List

from fastapi import APIRouter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from api.schemas import chroma as chroma_schemas
from api.settings import chroma as chroma_settings

settings = chroma_settings.ChromaSettings()
router = APIRouter()
chroma_db = Chroma(
    persist_directory=settings.persist_directory,
    embedding_function=OpenAIEmbeddings(
        model=settings.openai_model_name,
        deployment=settings.openai_deployment_id,
        openai_api_version=settings.openai_api_version,
        openai_api_base=settings.openai_api_base,
        openai_api_type=settings.openai_api_type,
        openai_api_key=settings.openai_api_key,
    ),
    collection_name=settings.collection_name,
)


@router.post("/chroma/search")
async def search(request: chroma_schemas.ChromaSearchRequest):
    docs = chroma_db.similarity_search_with_relevance_scores(request.query)
    results: List[chroma_schemas.Document] = []

    for index in range(min(len(docs), request.n_results)):
        results.append(
            chroma_schemas.Document(
                metadatas=chroma_schemas.Document.Metadatas(
                    source="",
                    title="",
                    score=docs[index][1],
                ),
                text=docs[index][0].page_content,
            )
        )
    return chroma_schemas.ChromaSearchResponse(
        query=request.query,
        id=str(uuid.uuid4()),  # FIXME: implement
        documents=results,
    )

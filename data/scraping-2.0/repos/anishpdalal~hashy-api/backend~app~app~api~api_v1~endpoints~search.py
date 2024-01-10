import os
import uuid

from fastapi import APIRouter, Depends
import gantry
import openai
import pinecone
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    current_active_user,
    get_async_session,
)
from app.crud.source import get_sources
from app.models.user import User
from app.schemas.search import Event, SearchResponse

PINECONE_KEY = os.getenv("PINECONE_KEY")
environment = os.getenv("ENVIRONMENT")
pinecone.init(api_key=PINECONE_KEY, environment="us-west1-gcp")
index = pinecone.Index(index_name="semantic-text-search")
openai.api_key = os.getenv("OPENAI_API_KEY")
search_model = SentenceTransformer("/mnt/bi_encoder")
gantry.init(
    api_key=os.getenv("GANTRY_API_KEY"),
    environment=os.getenv("ENVIRONMENT")
)

api_router = APIRouter()


@api_router.get("/search", tags=["search"], response_model=SearchResponse)
async def search(
    query: str,
    doc_type: str = None,
    user: User = Depends(current_active_user),
    db: AsyncSession = Depends(get_async_session),
    count: int = 10,
    log_id: str = None
):
    user_id = str(user.id)
    user_email = user.email
    source_ids = [str(source.id) for source in await get_sources(db, user_id, user_email)]
    filter = {"source_id": {"$in": source_ids}}
    if doc_type:
        filter["doc_type"] = {"$eq": doc_type}
    query_embedding = search_model.encode([query])
    query_results = index.query(
        queries=[query_embedding.tolist()],
        top_k=count,
        filter=filter,
        include_metadata=True,
        include_values=False,
        namespace=environment
    )
    matches = query_results["results"][0]["matches"]
    query_id = str(uuid.uuid4())
    results = {
        "query": query,
        "query_id": query_id,
        "count": len(matches),
        "results": [],
        "answer": None
    }
    for match in matches:
        metadata = match["metadata"]
        score = match["score"]
        if results["answer"] is None and len(matches) > 0:
            prompt = "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext:\n{0}\n\n---\n\nQuestion: {1}\nAnswer:"
            try:
                response = openai.Completion.create(
                    engine="text-curie-001",
                    prompt=prompt.format(metadata["text"], query),
                    temperature=0,
                    max_tokens=100,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                results["answer"] = response.choices[0]["text"].strip()
                if results["answer"].startswith("I don't know"):
                    results["answer"] = None
            except Exception as e:
                pass
        result = {
            "score": score,
            "doc_name": metadata["doc_name"],
            "doc_last_updated": str(metadata["doc_last_updated"]),
            "doc_url": metadata["doc_url"],
            "text": metadata["text"]
        }
        results["results"].append(result)

    if matches:
        gantry.log_record(
            application="search_endpoint",
            version=0,
            inputs={
                "query": query,
                "user_id": user_id,
                "log_id": log_id
            },
            outputs={
                "first_result_score": matches[0]["score"],
                "first_result_doc_name": matches[0]["metadata"]["doc_name"],
                "first_result_doc_url": matches[0]["metadata"]["doc_url"],
            },
            feedback_id={"id": query_id}
        )
    return results


@api_router.post("/log", tags=["search"])
async def log(event: Event, user: User = Depends(current_active_user)):
    gantry.log_record(
        application="search_endpoint",
        version=0,
        feedback_id={"id": event.query_id},
        feedback={"event_type": event.event_type, "message": event.message}
    )
    return {"message": "success"}
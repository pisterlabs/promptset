from langchain.chains import RetrievalQA
from fastapi import Request

from .prompts import PROMPT
from .model import get_chat_model
from .database import get_db


async def get_retrival_qa(request: Request):
    db = get_db()

    return RetrievalQA.from_chain_type(
        llm=get_chat_model(),
        chain_type="stuff",
        retriever=db.as_retriever(earch_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

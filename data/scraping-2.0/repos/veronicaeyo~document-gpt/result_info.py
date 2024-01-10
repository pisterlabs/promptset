from typing import List, TypedDict
from langchain.docstore.document import Document

class ResultInfo(TypedDict):
    question: str
    chat_history: List[List[str]]
    answer: str
    source_documents: List[Document]
    generated_question: str
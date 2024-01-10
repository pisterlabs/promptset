"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain

vectorstore: Optional[VectorStore] = None

def input_aggregator(question, metadata):
    template = f"{question}"
    if metadata["topic"] is not None:
        template += f" with topic: {metadata['topic']}"
    if metadata["context"] is not None:
        template += f" background of the question {metadata['context']}"
    if metadata["legalAct"] is not None:
        template += f" in legal act: {metadata['legalAct']}"
    if metadata["article"] is not None:
        template += f" in article: {metadata['article']}"
    if metadata["comDelegated"] is not None:
        template += f" with com delegated: {metadata['comDelegated']}"
    return template

logging.info("loading vectorstore")
if not Path("vectorstore.pkl").exists():
    raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
with open("sample_vectorstore.pkl", "rb") as k:
    sample_vectorstore = pickle.load(k)


question_handler = QuestionGenCallbackHandler([])
stream_handler = StreamingLLMCallbackHandler([])
qa_chain = get_chain(vectorstore, sample_vectorstore, question_handler, stream_handler)

question = """What is the meaning of the last sentence of Article 18(1) Regulation (EU) No 575/2013 (CRR)? Does it mean that the method of prudential consolidation (paragraphs 2 to 8) is not available for institutions that have to apply Part Six on the basis of their consolidated situation? When do institutions have to apply Part Six on the basis of their consolidated situation – is this only according to Article 11 of CRR or also in case of application for a liquidity sub-group according to Article 8(1)(a) of CRR?"""
metadata = {"topic": None, "context": None, "legalAct": None, "article": None, "comDelegated": None}
result = qa_chain._call(
                {"question": input_aggregator(question, metadata)}
            )
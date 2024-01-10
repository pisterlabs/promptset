from typing import List

import langchain
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from chatytt.chains.standard import ConversationalQAChain  # noqa:F401
from chatytt.chains.custom import (
    ConversationalQALCELChain,  # noqa:F401
    ConversationalQASequentialChain,  # noqa:F401
)
from chatytt.vector_store.pinecone_db import PineconeDB

langchain.debug = True


if __name__ == "__main__":
    load_dotenv()

    pinecone_db = PineconeDB(
        index_name="youtube-transcripts", embedding_source="open-ai"
    )

    # conversational_qa_chain = ConversationalQAChain(
    #     vector_store=pinecone_db.vector_store
    # )
    # conversational_qa_chain = ConversationalQALCELChain(
    #     vector_store=pinecone_db.vector_store,
    #     chat_model=ChatOpenAI(temperature=0.0)
    # )
    conversational_qa_chain = ConversationalQASequentialChain(
        vector_store=pinecone_db.vector_store, chat_model=ChatOpenAI(temperature=0.0)
    )

    query = (
        "Is buying a house a good financial decision to make in your 20s ? Give details on the "
        "reasoning behind your answer. Also provide the name of the speaker in the provided context from"
        "which you have constructed your answer."
    )

    chat_history: List[tuple[str, str]] = []
    response = conversational_qa_chain.get_response(
        query=query, chat_history=chat_history
    )
    print(response)

    chat_history = [(query, response)]
    query = "What is their opinion on what the right time to one is ?"
    response = conversational_qa_chain.get_response(
        query=query, chat_history=chat_history
    )
    print(response)

    chat_history.append((query, response))
    query = "Why do so many people recommend buying one ?"
    response = conversational_qa_chain.get_response(
        query=query, chat_history=chat_history
    )
    print(response)

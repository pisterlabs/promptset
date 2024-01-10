import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import KayAiRetriever


def get_response(query, configurations, chat_history):

    os.environ["KAY_API_KEY"] = configurations["kay_api_key"]

    model = ChatOpenAI(model="gpt-3.5-turbo-16k", openai_api_key=configurations["openai_api_key"])
    retriever = KayAiRetriever.create(dataset_id="company", data_types=["10-K", "10-Q"], num_contexts=6)
    qa = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever)

    result = qa({"question": query, "chat_history": chat_history})

    chat_history.append((query, result["answer"]))

    return result["answer"], chat_history

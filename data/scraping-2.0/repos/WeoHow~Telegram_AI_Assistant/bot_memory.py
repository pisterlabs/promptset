import os
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone

import pinecone


# initialize pinecone
load_dotenv()
pinecone.init(
    api_key = os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment = os.getenv("PINECONE_ENV")  # next to api key in console
)

def qa_memory(ask_question, llm):

    docs = [Document(page_content="qwerytp", metadata={"url": "None"})]
    embeddings = OpenAIEmbeddings()
    index_name = "telegram-chat-bot"
    vector_store = Pinecone.from_documents(documents=docs, embedding=embeddings, index_name=index_name)

    # the vector lookup still returns the semantically relevant information
    retriever = vector_store.as_retriever(search_kwargs=dict(k=4))
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    _DEFAULT_TEMPLATE = """下面是一段人類與AI助理的友好對話。AI很健談，並根據其上下文提供了許多具體細節。
    如果AI不知道問題的答案，AI會如實說不知道，不會編造不存在的資訊。
    
    歷史對話:
    {history}

    (如果和當前對話内容不相關，就不需要使用歷史對話的信息。)

    當前對話:
    人類: {input}
    AI助理:"""

    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
    )

    conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=memory,
        verbose=True
    )

    response = conversation_with_summary.predict(input=ask_question)

    return response
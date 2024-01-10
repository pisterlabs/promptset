from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.vectorstores.pinecone import Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()
pinecone.init(
    api_key="api-key",
    environment="env",
)
index = pinecone.Index('doc-convo-pod')
def run_llm(query:str,chat_history: list[dict[str,any]]=[]) -> any :

    embeddings= OpenAIEmbeddings()

    docsearch = Pinecone.from_existing_index(
        index_name="doc-convo-pod",
        embedding=embeddings
    )
    chat = ChatOpenAI(verbose = True, temperature = 0)

    # If we don't want any chat_history then uncomment below code
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    # )

    qa=ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever()
    )
    return qa({"question": query, "chat_history": chat_history})



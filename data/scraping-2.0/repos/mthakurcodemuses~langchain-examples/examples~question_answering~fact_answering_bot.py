from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from facts_custom_retriever import RedundantFilterRetriever
import langchain

# Load environment variables
load_dotenv()

langchain.debug = True

# Instantiate embeddings
open_ai_embeddings = OpenAIEmbeddings()
open_ai_chat_llm = ChatOpenAI()

# Instantiate ChromaDB to be used in retrieval
chroma_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=open_ai_embeddings)

# Instantiate custom retriever
embeddings_retriever = RedundantFilterRetriever(embeddings=open_ai_embeddings,
                                                chroma=chroma_db)

# Instantiate RetrievalQA chain
facts_retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=open_ai_chat_llm,
    retriever=embeddings_retriever,
    chain_type="stuff")

result = facts_retrieval_qa_chain.run("What is an interesting fact about the English language?")
print(result)

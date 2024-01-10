from pathlib import Path
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from consts import INDEX_NAME

if Path("./.env").is_file():
    from dotenv import load_dotenv  
    load_dotenv()

embedding = OpenAIEmbeddings()
vector_db = FAISS.load_local(INDEX_NAME, embedding)
llm = ChatOpenAI(verbose=True, temperature=0)
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_db.as_retriever(), return_source_documents=True)

def chat_with_model(question, chat_history):
    ans = qa({"question": question, "chat_history": chat_history})
    return ans

if __name__ == "__main__":
    chat_with_model()

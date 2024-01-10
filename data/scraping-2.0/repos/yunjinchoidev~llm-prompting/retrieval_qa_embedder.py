from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter, CharacterTextSplitter
from load_dotenv import load_dotenv
import os
from langchain.vectorstores import Pinecone
import pinecone
from langchain.chains import RetrievalQA

load_dotenv()
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

if __name__ == "__main__":
    loader = TextLoader("./embeded_document/naver_news.txt")
    document = loader.load()
    # print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    # print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="news")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    query = "what is vectordb"

    result = qa({"query": query})
    print(result)

from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import qdrant_client
import langchain
langchain.debug = True

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-MiniLM-L-6-v3")

client = qdrant_client.QdrantClient(
    host="localhost",
    prefer_grpc=False
)

qdrant = Qdrant(
    client=client,
    collection_name="aiw",
    embeddings=embeddings,
    metadata_payload_key="payload"
)

retriever = qdrant.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever)

question = "What does Alice drink?"
answer = qa.run(question)
print(answer)
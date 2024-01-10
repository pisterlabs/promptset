import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector

load_dotenv()

openai_key = os.getenv("OPENAI_KEY")
url = os.getenv("NEO4J_URL")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")

chat_llm = ChatOpenAI(openai_api_key=openai_key)

embedding_provider = OpenAIEmbeddings(openai_api_key=openai_key)

movie_plot_vector = Neo4jVector.from_existing_index(
    embedding_provider,
    url=url,
    username=user,
    password=password,
    index_name="moviePlots",
    embedding_node_property="embedding",
    text_node_property="plot",
)

retrievalQA = RetrievalQA.from_llm(
    llm=chat_llm,
    retriever=movie_plot_vector.as_retriever(),
    verbose=True,
    return_source_documents=True,
)

r = retrievalQA("A mission to the moon goes wrong")
print(r)
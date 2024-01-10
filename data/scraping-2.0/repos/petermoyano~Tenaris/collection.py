from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import qdrant_client
import os

# Initialize a Qdrant client. Returns an object used to interact with the Qdrant server
client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key= os.getenv("QDRANT_API_KEY")
)

#create a qdrant collection
def create_collection():
    os.environ['QDRANT_COLLECTION_NAME'] = "Tenaris"
    #create vectors configuration object
    vectors_config = qdrant_client.http.models.VectorParams(
        size=1536,
        distance=qdrant_client.http.models.Distance.COSINE
    )
    client.create_collection(
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        vectors_config=vectors_config
    )

#create vector store object
def add_texts_to_collection():
    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
            client=client,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            embeddings=embeddings
        )

    # Add documents to the cloud collection
    # get_chunks function takes a text and returns a list of chunks
    def get_chunks(text):
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
            )
        chunks = splitter.split_text(text)
        return chunks

    with open("data/paper.txt", "r") as f:
        raw_text = f.read()

    texts = get_chunks(raw_text)
    return vector_store.add_texts(texts)

# plug vector store into retrieval qa chain

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=Qdrant(
            client=client,
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            embeddings=OpenAIEmbeddings()
        ).as_retriever()

)
query = "What is the text about?"
print(qa.run(query))

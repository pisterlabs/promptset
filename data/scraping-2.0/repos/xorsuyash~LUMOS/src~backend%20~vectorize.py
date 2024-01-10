import langchain 
from langchain.vectorstores import Qdrant 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import dotenv
import os 
import qdrant_client
import sys 
from dataset import TextDataset 
import openai


#loading enviroment variables for quadrant 
dotenv.load_dotenv()
quadrant_host=os.getenv('QUADRANT_HOST')
quadrant_api_key=os.getenv('QUADRANT_API_KEY')


#seting up quadrant client 
client = qdrant_client.QdrantClient(
        quadrant_host,
        api_key=quadrant_api_key
    )

#collection name 
#os.environ["QDRANT_COLLECTION_NAME"] = "DSA-collection"

#creating_collection 
vector_config=qdrant_client.http.models.VectorParams(
    size=1536,
    distance=qdrant_client.http.models.Distance.COSINE
    
)

client.recreate_collection(
    collection_name="DSA-collection",
    vectors_config=vector_config,
)



#creating a vector store 
openai_api_key=os.getenv('OPENAI_API_KEY')
embeddings=OpenAIEmbeddings()
vectorstore=Qdrant(
    client=client,
    collection_name="DSA-collection",
    embeddings=embeddings
)

dataset=TextDataset('/home/xorsuyash/Desktop/LUMOS/GFG')
text_chunks=dataset.text_loader()
print("Chunks created")

vectorstore.add_texts(text_chunks[:100])



import pinecone
import os
from langchain.llms import Modal
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

pinecone.init(
	api_key=os.environ['PINECONE_API_KEY'],      
	environment='gcp-starter'      
)
model = Modal(endpoint_url=os.environ['MODAL_ENDPOINT_URL'])
INDEX_NAME = 'modal'

embeddings = OpenAIEmbeddings()
docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings)

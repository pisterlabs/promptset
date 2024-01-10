from flask import Blueprint, request, jsonify
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter

vectordb_blueprint = Blueprint('vectordb', __name__)

search_index = {}

@vectordb_blueprint.route('/vectordb', methods=['POST'])
def create_vectordb():
  # Get the documents from the request
  documents = request.json.get('documents', [])
  # Split the documents into chunks
  source_chunks = []
  splitter = CharacterTextSplitter(separator=" ",
                                   chunk_size=1024,
                                   chunk_overlap=0)
  for source in documents:
    for chunk in splitter.split_text(source.page_content):
      source_chunks.append(
        Document(page_content=chunk, metadata=source.metadata))
  # Create the vector database
  search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
  return jsonify({'message': 'Vector database created successfully'})


@vectordb_blueprint.route('/vectordb/query', methods=['POST'])
def query_vectordb():
  # Get the topic from the request
  topic = request.json.get('topic', '')
  # Find the documents in the vector index that correspond to the topic
  docs = search_index.similarity_search(topic, k=4)
  # Return the documents
  return jsonify([doc.page_content for doc in docs])


from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

import json
import config as cfg

class JsonImporter:
    def __init__(self):
        embeddings = OpenAIEmbeddings(
            openai_api_key=cfg.OPENAI_API_KEY,
            chunk_size=cfg.OPENAI_EMBEDDINGS_CHUNK_SIZE,
        )
      
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHARACTER_SPLITTER_CHUNK_SIZE,
            chunk_overlap=0,
        )
        
        self.db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
    def add_new_documents(self, texts, metadatas):
        docs = self.text_splitter.create_documents(texts, metadatas=metadatas)
        self.db.add_documents(docs)

    def add_documents_from_json(self, json_file):
        documents = json.loads(open(json_file).read())
        text_batch = []
        metadata_batch = []
        count = 1
        
        for document in documents:
          try:
            print(f'Processing document number {count}')
            if len(text_batch) < 100:
              text_batch.append(document['title'] + '. ' + document['body'])
              metadata_batch.append({'slug': document['slug'],'author': document.get('author', 'Le Trong Hoang Minh'), 'title': document['title']})
            else: 
              self.add_new_documents(texts=text_batch, metadatas=metadata_batch)
              text_batch = []
              metadata_batch = []
            count += 1
          except:
            continue

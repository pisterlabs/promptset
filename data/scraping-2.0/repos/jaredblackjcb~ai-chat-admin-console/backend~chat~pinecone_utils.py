import os
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ChatMessageHistory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from django.conf import settings

class PineconeUtils():
    
    def __init__(self, bot_id: str, namespace: str, context=None):
        self.namespace = bot_id + '-' + namespace
        self.context = context # optional conversation context
        self.index_name = os.environ.get('PINECONE_INDEX')
        pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'),
                    environment=os.environ.get('PINECONE_ENV'))
        
    def encode_documents(self, files):
        documents = []
        for file_obj in files:
            # loader = TextLoader(file)
            temp_file_path = os.path.join(settings.MEDIA_ROOT, file_obj.name)
            with open(temp_file_path, 'wb') as temp_file:
                for chunk in file_obj.chunks():
                    temp_file.write(chunk)
            # TODO: Add a loader method to determine the best way to load each file
            loader = PyPDFLoader(temp_file_path)
            documents += loader.load()
            
            # Remove the temporary file after processing
            os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            )
        chunks = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        # Generate document vectors and automatically upsert them into Pinecone
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=self.index_name, namespace=self.namespace)
        
    def get_reply(self, query):
        embeddings = OpenAIEmbeddings()
        vector_store = Pinecone.from_existing_index(index_name=self.index_name, embedding=embeddings, namespace=self.namespace)
        relevant_docs = vector_store.similarity_search(query)
        print(relevant_docs)

        llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        answer = chain.run(query)
        return answer
    
    def create_index(self, index_name):
        # Create a new Pinecone index
        if index_name not in pinecone.list_indexes():
            print(f"Creating index {index_name}")
            # OpenAI embeddings have a dimension of 1536
            pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x2")
            print("Done")
        else:
            print(f"Index {index_name} already exists")

    def delete_data_source(self, file_name):
        index = pinecone.Index(self.index_name)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)
        index.delete(namespace=self.namespace, filter={"source": file_path})

    def _generate_chat_history(self):
        history = ChatMessageHistory()
        for message in self.context:
            if message['type'] == 'bot':
                history.add_ai_message(message['message'])
            else:
                history.add_user_message(message['message'])
        return history

    def _get_relevant_context_data(self, query):
        return vector_store.similarity_search(query, namespace=self.namespace)


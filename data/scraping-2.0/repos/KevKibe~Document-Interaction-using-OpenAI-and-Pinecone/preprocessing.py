from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pinecone
import os
from dotenv import load_dotenv
from file_loader import DocumentLoader
from sklearn.decomposition import PCA

load_dotenv('.env')

openai_key = os.getenv('OPENAI_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('PINECONE_INDEX_NAME')


class Chunks:
    def get_chunks(self,documents):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function= len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks
    
class Vectorstore:    
    def get_vectorstore(self, chunks):

        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        doc_embeddings = []
        for text in chunks:
            doc_embedding =  embeddings.embed_documents(" ".join(str(text).split("\n")))
            doc_embeddings.append(doc_embedding)
        
        # reduce the dimensionality of the embeddings using PCA
        pca = PCA(n_components=25)
        reduced_embeddings = pca.fit_transform(doc_embeddings)
        vectorstore = Pinecone.from_documents(chunks, reduced_embeddings, index_name= 'index1')
        #vectorstore = vectorstore.to_numpy()
        return vectorstore

class ConversationChain:
    def get_conversation_chain(self, vectorstore):
        llm = ChatOpenAI(model_kwargs={'api_key': openai_key})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

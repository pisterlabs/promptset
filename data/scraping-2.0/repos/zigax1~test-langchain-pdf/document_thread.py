import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import uuid
from flask import jsonify
import boto3

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from chromadb.utils import embedding_functions

from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings

from langchain import PromptTemplate, LLMChain

from langchain.callbacks import get_openai_callback
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

VERBOSE = True

def getFileUrl(fileKey, bucketName):
    # Create an S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))


    # Get the URL of the uploaded file
    file_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucketName, 'Key': fileKey},
        ExpiresIn=3600  # URL expiration time in seconds
    )

    return file_url

class DocumentThread:
    def __init__(self, id = '', name = "", return_source = True, temperature = 0, model_name = "gpt-3.5-turbo", pdf_chunk_overlap = 200, pdf_chunk_size = 1000):
        self.id = id
        self.name = name
        self.documents = []
        self.return_source = return_source
        self.pdf_chunk_overlap = pdf_chunk_overlap
        self.pdf_chunk_size = pdf_chunk_size
        self.temperature = temperature
        self.model_name = model_name
        self.embedding = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'),model_name="text-embedding-ada-002")
        self.embeddingsOpenAi = OpenAIEmbeddings()
        self.chroma_client = chromadb.Client(Settings(chroma_api_impl="rest",chroma_server_host="localhost",chroma_server_http_port="8000"))
        self.vectordb = None
        self.vector_db_persist_directory = None
        self.llm = None
        self.memory = None
        self.chain = None
        self.collection = None
        self.getCollection = None

    def getInfo(self, collection_id):
        collection_name = "collection-" + str(collection_id)
        collection = self.chroma_client.get_collection(name=collection_name, embedding_function=self.embedding)
        return collection.get()
        
    def createCollection(self, collection_id):
        collection_name = "collection-" + str(collection_id)
        collection = self.chroma_client.create_collection(name=collection_name, embedding_function=self.embedding)
        # collection_dict = collection.to_dict()
        return collection.get()

    def loadFile(self, collection_id, fileKey, bucketName):
        collection_name = "collection-" + str(collection_id)
        # loader = UnstructuredPDFLoader(filepath)
        filepath = getFileUrl(fileKey, bucketName)
        print(filepath)
        loader = OnlinePDFLoader(filepath)
        document = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.pdf_chunk_size, chunk_overlap=self.pdf_chunk_overlap)
        document_objects = text_splitter.split_documents(document)

        # Extract text from each Document object
        documents = [doc.page_content for doc in document_objects]

        metadatas = []
        ids = []
        for i, doc in enumerate(document_objects):
                # tukaj se lahko za kasneje dajo recimo pixl teksta v pdf-ju, da se lahko prikaze to.. kuzner ma dobre ideje
                metadata = {"page": i + 1, "source": bucketName} 
                metadatas.append(metadata)
                doc_id = str(uuid.uuid4())  # Generate a random UUID for each document
                ids.append(doc_id)

        collection = self.chroma_client.get_collection(name=collection_name, embedding_function=self.embedding)
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        #  tu nekak pac izpisi imena pa to..
        print("Thread", collection_name, True if collection_name==collection_name else False)
        return True if collection_name==collection_name else False
        
    def askQuestion(self, collection_id, question):
        collection_name = "collection-" + str(collection_id)
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature, openai_api_key=os.environ.get('OPENAI_API_KEY'), streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,  output_key='answer')
        
        chroma_Vectorstore = Chroma(collection_name=collection_name, embedding_function=self.embeddingsOpenAi, client=self.chroma_client)

        # Check if the VectorStore has any documents
        collection = self.chroma_client.get_collection(name=collection_name, embedding_function=self.embedding)

        # if len(collection.get()['documents']) == 0:
        #         # Use LLMChain to call the basic model with the question and memory
        #         template = "Human: {input}\nAI:"

        #         prompt = PromptTemplate(template=template, input_variables=["input"])

        #         basic_model_chain = LLMChain(llm=self.llm, prompt=prompt)
        #         memory_variables = self.memory.load_memory_variables({"input": question})
        #         result = basic_model_chain.run({"input": question, **memory_variables})

        #         self.memory.save_context({"input": question}, {"outputs": result})

        #         return {
        #             "answer": result,
        #             "source_documents": []
        #         }
    
        self.chain = ConversationalRetrievalChain.from_llm(self.llm, chroma_Vectorstore.as_retriever(similarity_search_with_score=True),
                                                            return_source_documents=True,verbose=VERBOSE, 
                                                            memory=self.memory)
        
        
        result = self.chain({"question": question})
        # print(result)
        return result
        
        res_dict = {
            "answer": result["answer"],
        }

        res_dict["source_documents"] = []

        for source in result["source_documents"]:
            res_dict["source_documents"].append({
                "page_content": source.page_content,
                "metadata":  source.metadata
            })

        return res_dict
        


        



    # def getHistory(self):
    #     if self.memory is None:
    #         return []
    #     res = []
    #     for message in self.memory.chat_memory.messages:
    #         message_dict = message.dict()
    #         if isinstance(message, HumanMessage):
    #             message_dict["type"] = "Human"
    #         if isinstance(message, AIMessage):
    #             message_dict["type"] = "AI"
    #         res.append(message_dict)
    #     return res
    






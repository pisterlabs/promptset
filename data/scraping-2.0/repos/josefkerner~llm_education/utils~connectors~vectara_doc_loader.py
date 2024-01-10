from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.vectorstores import Vectara, VectorStore
import os
class DocumentLoader:
    def __init__(self):
        corpus_api_key = os.environ['VECTARA_API_KEY']
        customer_id = os.environ['VECTARA_CUSTOMER_ID']
        corpus_id = os.environ['VECTARA_CORPUS_ID']


        #Can be any vector store supported by langchain
        self.vector_store_client: VectorStore = Vectara(
            vectara_customer_id=customer_id, #ID of the customer
            vectara_api_key=corpus_api_key, #API key specific for the corpus
            vectara_corpus_id=corpus_id #Corpus ID
        )

    def add_doc_to_vector_store(self, document):
        '''
        Adds Langchain based Document object to Vectara vector store
        :param document:
        :return:
        '''
        try:
            self.vector_store_client.add_documents([document])
        except Exception as e:
            raise ValueError("Failed to add to vector store")
    def load_doc(self, doc_path:str):
        '''
        Loads a document from a path
        :param doc_path:
        :return:
        '''
        if doc_path.endswith('.pdf'):
            loader = PyPDFLoader(doc_path)
        elif doc_path.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(doc_path)
        elif doc_path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(doc_path)
        else:
            loader = UnstructuredHTMLLoader(doc_path)
        return loader.load()[0]
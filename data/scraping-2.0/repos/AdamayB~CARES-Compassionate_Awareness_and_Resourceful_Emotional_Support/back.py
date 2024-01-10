from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

dataPath='Data/'
FaissPath = 'vectorstore/db_faiss'

def createVectorDB():
    loader = DirectoryLoader(dataPath,glob='*.pdf',loader_cls=PyPDFLoader)

    documents = loader.load()
    textSplit = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    texts = textSplit.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device':'cpu'}) # Check Line 26

    dataBase =  FAISS.from_documents(texts,embeddings)
    dataBase.save_local(FaissPath)

if __name__=='__main__':
    createVectorDB()

'''
Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl,
 ideep, hip, ve, fpga, ort, xla, 
lazy, vulkan, mps, meta, hpu, mtia, privateuseone 
'''

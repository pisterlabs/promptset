from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
PALM_API_KEY="AIzaSyC9C5e3RlBDGxMUS79VdXcyE4FapZ4EnQM"


def pdfqna(query):
    
    files_path = "data.pdf"
    loaders = [UnstructuredPDFLoader(files_path)]

    index = VectorstoreIndexCreator(
        embedding=GooglePalmEmbeddings(google_api_key=PALM_API_KEY),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0),
    ).from_loaders(loaders)


    llm = GooglePalm(temperature=0.1,google_api_key=PALM_API_KEY)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        # input_key="question",
        return_source_documents=True,
        )
    
    response = chain(query)
    return response["result"]

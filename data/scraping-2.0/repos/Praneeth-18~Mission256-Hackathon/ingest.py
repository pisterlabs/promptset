from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders.csv_loader import CSVLoader
import os
DATA_PATH = 'Data/'
DB_FAISS_PATH = 'vectorstore/faiss_db'

# Create vector database
def create_vector_db():
    # loader = DirectoryLoader(DATA_PATH,
    #                          glob='*.pdf',
    #                          loader_cls=PyPDFLoader)

    
    
    # loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
    #             'delimiter': ','})
    

    # data = loader.load()

    loaders = {
    '.pdf': PyPDFLoader,
    '.csv': CSVLoader,
    }

# Define a function to create a DirectoryLoader for a specific file type
    def create_directory_loader(file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
        )   

    # Create DirectoryLoader instances for each file type
    pdf_loader = create_directory_loader('.pdf',DATA_PATH)
    csv_loader = create_directory_loader('.csv', DATA_PATH)

    # Load the files
    pdf_documents = pdf_loader.load()

    csv_documents = csv_loader.load()

    # for root, dirs, files in os.walk("Data"):
    #     for file in files:
    #         if file.endswith(".pdf"):
    #             print(file)
    #             loader = PyPDFLoader(os.path.join(root, file))
    #             documents = loader.load()
    #         elif file.endswith(".csv"):
    #             print(file)
    #             loader = CSVLoader(file_path=os.path.join(root, file), encoding="utf-8", csv_args={
    #             'delimiter': ','})
    #             documents=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    pdftexts = text_splitter.split_documents(pdf_documents)
    csvtexts = text_splitter.split_documents(csv_documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db1 = FAISS.from_documents(pdftexts, embeddings)
    db2 = FAISS.from_documents(csvtexts, embeddings)
    db1.merge_from(db2)
    db1.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
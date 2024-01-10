from langchain.document_loaders import PyPDFLoader
import os

# Set 'directory' with the actual path of the directory you want to list
def collect_documents_from_directory(directory: str):
    file_names_list = os.listdir(directory)
    documents = []
    for file_name in file_names_list:
        print(file_name)
        pdf_path = os.path.join(directory,
                            file_name)

        loader = PyPDFLoader(pdf_path)
        document = loader.load()
        documents += document

    return documents
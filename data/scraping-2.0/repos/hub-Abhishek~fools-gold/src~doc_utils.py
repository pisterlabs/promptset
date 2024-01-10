import os
from src.utils import print_message
from PyPDF2 import PdfReader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.docstore.document import Document
import tempfile


from pydantic import Field
class Document:
    def __init__(self, page_content: str, metadata: dict = Field(default_factory=dict)) -> None:
        """Class for storing a piece of text and associated metadata."""
        """String text."""
        self.page_content = page_content
        """Dictionary of metadata."""
        self.metadata = metadata
        

def process_pdf(file, documents):
    pdfReader = PdfReader(file)
    # message = f'Number of pages in the uploaded pdf {file.name} - {len(pdfReader.pages)}'
    # print_message(message, st=st)
    
    for i, page in enumerate(pdfReader.pages):
        documents.append(Document(page_content=page.extract_text(),
                                metadata={'title': file, 
                                            'page_number': i,
                                            'file_name': file}))
    return documents

def process_text(file, documents):
    text = ''
    for line in file:
        text +=  str(line.decode())
    documents.append(Document(page_content=text,
                            metadata={'title': file,
                                      'file_name': file}))
    return documents

def process_csv(file, documents):
    if file :
    #use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file_path = tmp_file.name

        # import pdb; pdb.set_trace()
        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                    'delimiter': ','})
        data = loader.load()
        return data

def get_chunks(documents):
        
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator=' ')
    chunks = char_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        if len(chunk.page_content)>1000:
            chunks[i].page_content = chunk.page_content[:1000]
    return chunks


def process_file_get_chunks(uploaded_file):
    """
    1. Check for new files
    2. Fow new files - 
        1. Read the files
        2. Create documents
        3. Create chunks
        4. Return chunks
    """
    documents = []
    if uploaded_file[-4:]=='.pdf': 
        documents = process_pdf(uploaded_file, documents)
    
    if uploaded_file[-4:]=='.txt':
        documents = process_text(uploaded_file, documents)

    chunks = get_chunks(documents)
    return chunks





# def check_for_files(st):
#     if not os.path.exists('app_database'):
#         os.mkdir('app_database')
#         message = 'Created new folder - app_database'
#         print_message(message, st=st)
#     if not os.path.exists('app_database/db'):
#         os.mkdir('app_database/db')
#         message = 'Created new folder - app_database/db'
#         print_message(message, st=st)
#     if not os.path.exists('app_database/files.txt'):
#         with open('app_database/files.txt','w') as f:
#             message = 'Created new file - files.txt'
#             print_message(message, st=st)
#             f.close()
#     if not os.path.exists('app_database/result.txt'):
#         with open('app_database/result.txt','w') as f:
#             message = 'Created new file - result.txt'
#             print_message(message, st=st)
#             f.close()
#     if not os.path.exists('app_database/query_ids.txt'):
#         with open('app_database/query_ids.txt','w') as f:
#             message = 'Created new file - query_ids.txt'
#             print_message(message, st=st)
#             f.close()

# def check_for_new_files(files, st):
#     check_for_files(st)
#     files = list(set(files))
#     old_files = read_old_file_names()
#     if old_files==files:
#         message = 'No new files uploaded. Using the existing database.'
#         print_message(message, st=st)
#         return False
#     else:
#         write_new_file_names(files)
#         return True


# def read_old_file_names():
#     with open('app_database/files.txt','r') as f:
#         files = f.readlines()
#         f.close()
#     files = [file.strip() for file in files]
#     return files

# def write_new_file_names(files):
#     with open('app_database/files.txt','w') as f:
#         for item in files:
#             f.write(item+"\n")
#         f.close()
#     return None
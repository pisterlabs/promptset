from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader
from API_keys import OPENAI_API_KEY, PINECONE_API, PINECONE_ENV
import pinecone 
import os

with open('history.txt', 'r') as f:
    files_history = f.read().splitlines()

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)


# initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)

# initialize vectorstore
vectorstore = Pinecone(index=pinecone.Index("test"), embedding_function=embeddings.embed_query, text_key='text')

class PdfLoader:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_names = self.get_file_names()
       
    def get_file_names(self) -> list:
        """Returns a list of the names of all files in the specified folder."""

        names = []
        for file_name in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, file_name)):
                names.append(file_name)
        return names
    
    def load_files_in_pinecone(self, index_name: str) -> Pinecone:
            """Loads the pdf files and embeds it, then uploads it to pinecone and returns the docsearch object."""
     
            for file in self.file_names:

                all_files_already_uploaded = True

                if file not in files_history:

                    all_files_already_uploaded = False 

                    with open('history.txt', 'a') as fi:
                        fi.write(f'\n{file}')

                    loader = PyPDFLoader(f'files/{file}')
                    documents = loader.load()
                    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    docs = text_splitter.split_documents(documents)
                    vectorstore.from_documents(docs, embeddings, index_name=index_name)
                    print(f'{file} uploaded in PINECONE')
                    print('---------------------')
                
            if all_files_already_uploaded:
                print('All files already uploaded in PINECONE')
                print('---------------------')





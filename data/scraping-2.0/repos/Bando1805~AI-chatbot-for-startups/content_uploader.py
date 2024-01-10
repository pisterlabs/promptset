from langchain.vectorstores import Pinecone
from API_keys import OPENAI_API_KEY, PINECONE_API, PINECONE_ENV
import pinecone 
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.schema import Document

class CustomTextLoader:

    def __init__(self,  
                 folder_path,    
                 pinecone_api = PINECONE_API,   
                 pinecone_env = PINECONE_ENV,
                 ):
        
        self.pinecone_api = pinecone_api
        self.pinecone_env = pinecone_env
        self.initialize_pinecone()

        self.pinecone_index_name = 'test'
        self.embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)
        self.vectorstore = Pinecone(index=pinecone.Index(self.pinecone_index_name),
                                    embedding_function=self.embeddings.embed_query, 
                                    text_key='text',)

        self.folder_path = folder_path
        self.file_names = self.get_file_names()
        
    
    def initialize_pinecone(self) -> None:
        pinecone.init(
            api_key= self.pinecone_api,  # find at app.pinecone.io
            environment= self.pinecone_env  # next to api key in console
        )
    
       
    def get_file_names(self) -> list:
        """Returns a list of the names of all files in the specified folder."""
        names = []
        for file_name in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, file_name)):
                names.append(file_name)
        return names


    def load_files_in_pinecone(self) -> Pinecone:
        """Loads the pdf files and embeds it, then uploads it to pinecone and returns the docsearch object."""
            
        for file in self.file_names:
            # This is a long document we can split up.
            with open(f'files/{file}') as f:
                file_content = f.read()

            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_text(file_content)
            docs = [Document(page_content=text,metadata = {'source':f'{i}'}) for i,text in enumerate(texts)]
            self.vectorstore.from_documents(docs, self.embeddings, index_name = self.pinecone_index_name)
            print(f'{file} uploaded in PINECONE')
            print('---------------------')


txt_load = CustomTextLoader(folder_path = 'files')
txt_load.load_files_in_pinecone()

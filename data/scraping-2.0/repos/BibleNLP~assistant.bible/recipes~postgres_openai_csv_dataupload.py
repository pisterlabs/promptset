'''Inputs text data into vector db.
Content type: CSV/TSV with following header(id,text,label,links,medialinks)
File processor: the Common CSV processor inherited into the LangchainLoader class
Embedding: OpenAI ada embedding
DB: postgres with pgvector extension

prerequest: A postgres running o localhost:5434, with DB created
'''

import os
import glob
import sys
 
# setting path
sys.path.append('../app')

from core.pipeline import DataUploadPipeline
from core.vectordb.postgres4langchain import Postgres
from core.embedding.openai import OpenAIEmbedding
import schema

######## Configure the pipeline's tech stack ############
data_stack = DataUploadPipeline(
    vectordb=Postgres(
        host_n_port="localhost:5435",       # change if your port is different
        collection_name='adotbcollection',  # change if you db name is different
        user='admin',
        password="secret",
        embedding=OpenAIEmbedding(
            api_key=os.getenv('OPENAI_API_KEY'),
            model='text-embedding-ada-002')
        ),   
)

######## File Processor #############
INPUTFILE = "./data/dataupload.tsv"

processed_documents = data_stack.file_processor.process_file(
        file=INPUTFILE,
        file_type=schema.FileType.CSV,
        col_delimiter="\t")
print(f"Created {len(processed_documents)} documents")
print('One Sample Document: ', processed_documents[0], '\n\n')



############# Embeddings  ###############
data_stack.embedding.get_embeddings(doc_list=processed_documents)
# print('One Sample Document(with embeddings): ', processed_documents[0], '\n\n')


# ########### Adding to postgres DB #################
data_stack.vectordb.add_to_collection(docs=processed_documents[:10])
print("check the embeddings table of your DB for new rows")
print("!!!!!!!!!!!!!! Finished !!!!!!!!!!!!!!!!")

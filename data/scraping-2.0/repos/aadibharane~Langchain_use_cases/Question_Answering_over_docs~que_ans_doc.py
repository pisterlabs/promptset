#Question Answering
'''
This notebook walks through how to use LangChain for question answering over a list of documents. 
It covers four different types of chains: stuff, map_reduce, refine, map_rerank.
'''
#Prepare Data
'''
First we prepare the data. For this example we do similarity search over a vector database, but these documents could be 
fetched in any manner (the point of this notebook to highlight what to do AFTER you fetch the documents).
'''
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
import os
os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"  
def que_ans_doc():
    with open("E:\langchain\Question_Answering_over_docs\state_of_the_union.txt",encoding='utf-8') as f:
        state_of_the_union = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(state_of_the_union)

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()
    print(docsearch)

que_ans_doc()
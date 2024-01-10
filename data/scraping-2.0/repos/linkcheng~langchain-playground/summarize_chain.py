import os
from dotenv import load_dotenv

from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback

load_dotenv()

# 文本摘要

# define a function to load pdf
def load_pdf(path):
    loader = UnstructuredFileLoader(path)
    doc = loader.load()
    return doc


def split_text(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_doc = text_splitter.split_documents(doc)
    return split_doc
    

def load_chain(key):
    llm = OpenAI(openai_api_key=key, temperature=0)
    chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True)
    return chain

# main
if __name__ == '__main__':
    p = '~/Desktop/2306.02707.pdf'
    key = os.getenv("OPENAI_API_KEY")
    doc = load_pdf(p)
    print(f"{len(doc)=}, {len(doc[0].page_content)=}")
    split_docs = split_text(doc)
    print(f"{len(split_docs)=}")
    
    chain = load_chain(key)
    input_docs = split_docs[:20]
    
    with get_openai_callback() as cb:
        res = chain.run(input_documents=input_docs)
        print(res)
        print(f"{cb}")

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from .dto import PDFQuery
import uuid


def read_pdf(pdf):
    reader = PdfReader(pdf)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap = 200,
        length_function= len
    )
    texts = text_splitter.split_text(raw_text)
    
    embeddings = OpenAIEmbeddings()
    
    docsearch = FAISS.from_texts(texts, embeddings)
    
    id = str(uuid.uuid4())

    docsearch.save_local("data/" + id)


    return id

def chat_pdf(id,query):
    
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local("data/" + id, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    docs = docsearch.similarity_search(query)
    summary = chain.run(input_documents=docs, question=query)
    
    return summary
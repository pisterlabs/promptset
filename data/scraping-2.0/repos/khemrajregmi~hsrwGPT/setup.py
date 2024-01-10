import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI


def pre_setup():
    print('inside pre_setup')
    # os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = "your key here"
    doc_reader = PdfReader('data/QA.pdf')

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,  # striding over the text
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(),
                          chain_type="stuff")
    # we are going to stuff all the docs in at once
    print('before return')
    return docsearch, chain

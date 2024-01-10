from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from pathlib import Path


class HRBot:
    
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))
        raw_text = ''

        pdf_search = Path("./assets/").glob("*.pdf")
        pdf_files = [str(file.absolute()) for file in pdf_search]

        for pdf_file in pdf_files:
            reader = PdfReader(pdf_file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text

        text_splitter = CharacterTextSplitter(        
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )   
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings()
        self.docsearch = FAISS.from_texts(texts, embeddings)
        self.chain = load_qa_chain(OpenAI(), chain_type="stuff")
        

    def getAnswer(self, query):
        docs = self.docsearch.similarity_search(query)
        answer = self.chain.run(input_documents=docs, question=query)
        return answer
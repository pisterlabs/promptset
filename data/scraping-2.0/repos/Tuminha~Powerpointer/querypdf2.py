import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import pinecone
from altair import Chart
from io import BytesIO
import PyPDF2


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '4d940a59-e43b-4147-9dfe-5362b5ae7f80')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-east-1-aws') # You may need to switch with your env
index_name = "periobot"


st.title('PDF Query')

uploaded_files = [st.file_uploader(f"Upload PDF {i+1}", type=['pdf']) for i in range(4)]
query = st.text_input('Enter your query:')

if st.button('Run Query'):
    if not any(uploaded_files):
        st.write("Please upload at least one PDF file.")
    elif not query:
        st.write("Please enter a query.")
    else:
        data = []
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                reader = PyPDF2.PdfReader(BytesIO(uploaded_file.getvalue()))
                for page in range(len(reader.pages)):
                    data.append(Document(page_content=reader.pages[page].extract_text()))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

        docs = docsearch.similarity_search(query)

        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")

        result = chain.run(input_documents=docs, question=query)
        st.write(result)
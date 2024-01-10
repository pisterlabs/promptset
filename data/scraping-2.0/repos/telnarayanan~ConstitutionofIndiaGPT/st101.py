# List of Imports
import config
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA


os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
with open("COI_English.txt", "r", encoding="utf8") as f:
    constitution_text = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(constitution_text)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
    #query = "What is this document about? When was this document last updated?"

st.title('Understand Finance Minister\'s budget speech 2023-2024')

st.write('This webpage is designed to query FM\'s Budget Speech using Open AI\'s GPT-3 language processing technology. The page will allow users to search through the speech and find relevant information quickly and easily. With GPT-3\'s advanced natural language processing, users will be able to easily find exactly what they are looking for in the speech. Additionally, the page will provide users with a comprehensive summary of the contents of the speech so they can quickly get an overview of the main points.') 

st.write('Please navigate to [link](https://www.indiabudget.gov.in/doc/budget_speech.pdf) for the full text of the speech')

query = st.text_input('Enter your query in this text box about FM\'s 2023-2024 Budget speech','What are the differences in tax ceilings this year for salaried citizens?')
st.write(qa.run(query))

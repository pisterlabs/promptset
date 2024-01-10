import streamlit as st
import os
import toml
from langchain.chat_models import AzureChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader

with open('../secrets.toml', 'r') as f:
    config = toml.load(f)

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] =  config['OPENAI_API_BASE']
os.environ["OPENAI_API_KEY"] = config['OPENAI_API_KEY']
os.environ["OPENAI_API_VERSION"] = "2023-05-15"

DEPLOYMENT_NAME = "gpt-4-32k"
model = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"] ,
    openai_api_version="2023-05-15",
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
)

llm = AzureChatOpenAI(deployment_name="gpt-4-32k")
embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002", model="text-embedding-ada-002", chunk_size=1) # 


st.set_page_config(
    page_title="Home",
    page_icon="👨‍⚕️",
)

st.header("Welcome to Medical Smart Search👨‍⚕️")

with st.sidebar.expander(" 🛠️ Settings ", expanded=False):
    
    # FILE = st.selectbox(label='File', options=['./data/medical_paper.pdf', './data/mitochondrial_forms_of_diabetes.pdf', './data/Neuropsychiatric_symptoms.pdf']) 
    pdf_files = [f'./data/{file}' for file in os.listdir('./data') if file.endswith('.pdf')]
    FILE = st.sidebar.selectbox(label='File', options=pdf_files)

def get_answer(index, query):
    """Returns answer to a query using langchain QA chain"""

    docs = index.similarity_search(query)

    chain = load_qa_chain(llm)
    answer = chain.run(input_documents=docs, question=query)

    return answer

if FILE:
    loader = PyPDFLoader(FILE)
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, embeddings)

query = st.text_area("Ask a question about the document")

if query:
    
    docs = faiss_index.similarity_search(query, k=1)
    button = st.button("Submit")
    if button:
        st.write(get_answer(faiss_index, query))
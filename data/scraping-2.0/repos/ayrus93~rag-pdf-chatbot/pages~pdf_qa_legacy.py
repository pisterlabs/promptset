from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback



openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

def get_pdf_text(pdf_files):
    
    text = ""

    for pdf_file in pdf_files:
        print(pdf_file.name)
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()

    return text

def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )

    chunks = text_splitter.split_text(text)
    #print(chunks)
    return chunks


def get_vector_store(text_chunks):
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    
    return vectorstore

@st.cache_data
def process_files(uploaded_files):

    raw_text = get_pdf_text(uploaded_files)
    text_chunks = get_chunk_text(raw_text)
    vector_store = get_vector_store(text_chunks)

    return vector_store

@st.cache_data
def openai_query(query,query_type):

    docs = vector_store.similarity_search(query,k=2)
    #st.write(len(docs))
    #for doc in docs:
        #st.write(doc)
        #results = db.query(query_texts=[query],n_results=1)
    llm = OpenAI(openai_api_key=openai_api_key,model_name="gpt-3.5-turbo", temperature=0)
    chain = load_qa_chain(llm, chain_type='stuff')
        
    with get_openai_callback() as cost:
        if query_type == 'general':
            response = chain.run(input_documents=docs, question=query)
        elif query_type == 'detailed':
            response = chain.run(input_documents=docs, question=" If possible give a detailed description or a step by step instruction - " + query)
        print(cost)
    
    return response


st.title("Chat with your PDF 💬")



uploaded_files = st.file_uploader('Upload your PDF Documents', type='pdf',accept_multiple_files=True)



if uploaded_files:

    vector_store = process_files(uploaded_files)

    option = st.selectbox(
    'Please select your role below to chat',
    ('Investor', 'Clearing member', 'Trading member'))

    st.write('Hello ' + option + ', Please enter your query below')
    
    query = st.text_input('Ask a question to the PDF')
    cancel_button = st.button('Cancel',key="cancel")
    
    if cancel_button:
        st.stop()
    
    if query:
            query_type = 'general'
            response = openai_query(query,query_type)
            st.write(response)

            #tell_more_button = st.button("Tell me more",key="tell")
            col1,col2,col3 = st.columns([1,0.5,5])
            with col1:
                tell_more_button = st.button("Tell me more",key="tell")
            with col2:
                like_button = st.button("👍",key="like")
            with col3:
                dislike_button = st.button("👎",key="dislike")


            if tell_more_button:
                #results = db.query(query_texts=[query],n_results=1)
                query_type = 'detailed'
                response = openai_query(query,query_type)
                st.write(response)
    
            
            
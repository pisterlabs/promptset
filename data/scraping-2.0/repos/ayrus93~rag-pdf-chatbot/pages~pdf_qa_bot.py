from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import os
import tempfile
import datetime
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document

st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

#https://stackoverflow.com/questions/64719918/how-to-write-streamlit-uploadedfile-to-temporary-directory-with-original-filenam

#@st.cache_data
def openai_query(splits,query,query_type):

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(splits,embeddings)
    docs = vectordb.similarity_search_with_score(query,k=3)
    docs_with_score = []
    for doc, score in docs:
        #st.write(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
        doc.metadata['score'] = score
        n_doc =  Document(page_content=doc.page_content, metadata=doc.metadata)
        docs_with_score.append(doc)

    llm = OpenAI(openai_api_key=openai_api_key,model_name="gpt-3.5-turbo", temperature=0)
    chain = load_qa_chain(llm, chain_type='stuff')
        
    with get_openai_callback() as cost:
        if query_type == 'general':
            response = chain.run(input_documents=docs_with_score, question=query)
        elif query_type == 'detailed':
            response = chain.run(input_documents=docs_with_score, question="If possible give a detailed description or a step by step instruction - " + query)
        print(cost)
    
    return response,docs_with_score

def get_chunk_text(docs):
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    splits = text_splitter.split_documents(docs)
    return splits


files = st.file_uploader('Upload your PDF Documents', type='pdf',accept_multiple_files=True)

if files:


    loaders = []
    docs = []
    for i in range(len(files)):
        bytes_data = files[i].read()  # read the content of the file in binary
        #print(files[i].name, bytes_data)
        with tempfile.TemporaryDirectory() as directory_name:
            filename = os.path.join(directory_name, files[i].name)
            with open(filename, "wb") as f:
                f.write(bytes_data)
            loader = PyPDFLoader(filename)
            docs.extend(loader.load())
        

    for doc in docs:
        current_ts = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        doc.metadata['upload_ts'] = current_ts
        doc.metadata['page'] = doc.metadata['page'] + 1

    splits = get_chunk_text(docs)
    splits = docs
    

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
        ai_resp,docs_with_score = openai_query(splits,query,query_type)
        st.write(ai_resp)

        col1,col2,col3 = st.columns([1,0.5,5])
        with col1:
            tell_more_button = st.button("Tell me more",key="tell")
        with col2:
            like_button = st.button("👍",key="like",disabled=True)
        with col3:
            dislike_button = st.button("👎",key="dislike",disabled=True)


        if tell_more_button:
                #results = db.query(query_texts=[query],n_results=1)
            query_type = 'detailed'
            response,docs_with_score = openai_query(splits,query,query_type)
            st.write(response)

        tab_labels = []
        for i in range(len(docs_with_score)):
            i += 1
            name = "Source "+str(i)
            tab_labels.append(name)
        #tab1, tab2 = st.tabs(["Response 1", "Response 2"])

        tabs = st.tabs(tab_labels)
        for label, tab in zip(tab_labels, tabs):
            ind = tab_labels.index(label)
            with tab:
                #st.write(doc_content[ind])
                #st.write(doc_metadata[ind])
                #st.write(doc_score[ind])

                #st.write(docs[ind].page_content)
                st.write(docs_with_score[ind].page_content)
                st.write(docs_with_score[ind].metadata)
                #st.write(docs_with_score[ind])




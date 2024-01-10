import streamlit as st
from PyPDF2 import PdfReader
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import time
import os

## Have to add the hugging face api token in the .bashrc file for this to work

def genai(retriever, query, context):
    #repo_id = 'lmsys/fastchat-t5-3b-v1.0'
    repo_id = 'google/flan-t5-large'
    llm = HuggingFaceHub(huggingfacehub_api_token=os.getenv("HF_HOME_TOKEN"),
                         repo_id=repo_id,
                         model_kwargs={'temperature': 1e-10, 'max_length': 1000})
    retrieval_chain = RetrievalQA.from_chain_type(llm,chain_type='stuff',retriever=retriever.as_retriever())
    response = retrieval_chain.run(query).strip()
    return response
def main():
    st.title("Upload Your PDF Document.")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        embeddings = HuggingFaceEmbeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)
        docs = splitter.split_text(text)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                doc_search = pickle.load(f)
        else:
            doc_search = FAISS.from_texts(docs, embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(doc_search,f)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar", "👤")):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("How can I help you?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="👩‍💻"):
                st.markdown(prompt)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                context = doc_search.similarity_search(prompt, k=5)
                assistant_response = genai(doc_search, prompt, context)
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            if st.button('Clear Chat'):
                st.session_state.messages = []

if __name__ == '__main__':
    main()
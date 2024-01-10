import streamlit as st 
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import tempfile

DB_FAISS_PATH= "vectorstore/db_faiss"

#loading models
def load_llm():
    llm= CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temprature=0.5
    )
    return llm
#Creating User interface and uploader using streamlit
st.title("Chat with CSV using LLAMA 2 👩‍💻")
st.markdown("<h3 style='text-align: center; color: white;'> Built By <a href='https://www.linkedin.com//in//akash-kesrwani-45203017a'> Artificial Intelligence-Akash</a></h3>",
            unsafe_allow_html=True)
uploaded_file=st.sidebar._file_uploader("Upload Your CSV File", type="csv")

#we need our temporary local location 
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path=tmp_file.name

    loader=CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
        'delimiter':','
    })
    #loading data with the langchain functions
    data=loader.load()
    #st.json(data)
    #creating embeddings of data using sentence tranformer available on huggingface
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-l6-V2",
        model_kwargs={'device':'cpu'}
    )
    #connecting vector data store faiss to store the embeddings
    db=FAISS.from_documents(data, embeddings)
    db.save_local(DB_FAISS_PATH)
    llm=load_llm()
    #creating chain 
    chain=ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result=chain({'question': query, 'chat_history': st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    if 'history' not in st.session_state:
        st.session_state['history']=[]

    if 'generated' not in st.session_state:
        st.session_state['generated']=["Hello Buddy, ask me anything about "+ uploaded_file.name+" ! 🤗"]
    if "past" not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    #Container for the chat history
    response_container=st.container()

    #container for user's text input
    container=st.container()
    with container:
        with st.form(key="my_form", clear_on_submit=True):
            user_input=st.text_input("Query:", placeholder="Talk to your csv data (:",
                                     key='input')
            submit_button = st.form_submit_button(label='send')
        if submit_button and user_input:
            output=conversational_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user',
                            avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")


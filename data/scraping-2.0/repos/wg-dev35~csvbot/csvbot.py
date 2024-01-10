
#imports
import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

#Vector DB storage
DATA_PATH = "E:/LLMS/data/Recipie-db"
DB_FAISS_PATH = "E:/LLMS/vectorstores/db_faiss"

#model loading
def load_llm():
    llm = CTransformers(
        model = 'E:/LLMS/mistral-7b-instruct-v0.1.Q8_0.gguf',
        model_type = "mistral",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#ui
st.title("CSV chat using mistral 7b")
st.markdown("<h1 style='text-align: center; color: blue;'>CSV Bot with mistral</h1> :atom_symbol:", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Repurposed by <a href='https://github.com/wg-dev35'>Elgooey</a> for learning and developing skills</h3>", unsafe_allow_html=True)

#file handling
user_upload = st.sidebar.file_uploader("Upload file...", type="csv")

#ui logic
if user_upload:
    st.write("file uloaded") #debug
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(user_upload.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    embeds = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2', model_kwargs = {'device':'cpu'})##embeds inputs using cpu

    db = FAISS.from_documents(data, embeds)
    db.save_local(DB_FAISS_PATH)
    chain = ConversationalRetrievalChain.from_llm(llm=load_llm(), retriever=db.as_retriever())

    #chat logic
    def chats(query):
        result = chain({"question": query,"chat_history":st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result['answer']

    #initializing chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, what can i get for you from " + user_upload.name + ":hugs:"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey :wave:"]


    #chat history 
    hist_container = st.container()
    #user history
    container = st.container()

    with container:
        with st.form(key="test_form", clear_on_submit=True):
            user_input = st.text_input("Query:",  placeholder="Retreive from file", key='input')
            submitbtn = st.form_submit_button(label="Chat")
        if submitbtn and user_input:
            output = chats(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with hist_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state['generated'][i], key=str(i),avatar_style="thumbs")
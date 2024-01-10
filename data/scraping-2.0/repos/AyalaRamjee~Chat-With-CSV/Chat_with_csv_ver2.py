import streamlit as st 
from streamlit_chat import message
import tempfile
import numpy as np
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

DB_FAISS_PATH = 'vectorstore/db_faiss'

@st.cache(allow_output_mutation=True)
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

st.title("Chat with CSV using Llama2 🦙🦜")
st.markdown("<h3 style='text-align: center; color: black;'>Built by Ramjee - You're Welcome</h3>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

def generate_embeddings_for_batch(batch_data, embeddings_model):
    # Generate embeddings directly using the embeddings model
    embeddings = embeddings_model.embed_documents(batch_data)
    return embeddings

if uploaded_file:
    st.info("CSV file uploaded.")
    
    # Extracting data
    st.info("Extracting data from the file...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    st.success("Data extracted successfully!")
    
    # Try to infer column names from the first Document object
    if data and hasattr(data[0], '__dict__'):
        column_names = list(data[0].__dict__.keys())
    else:
        st.error("Unable to extract column names from the uploaded CSV.")
    
    # Allow users to select a column
    column_to_use = st.selectbox('Choose the column you want to chat about:', column_names)

    
    # Generating embeddings
    st.info("Generating embeddings for the selected column... This might take some time.")
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    
    # Partition the data into smaller batches
    BATCH_SIZE = 512  # Adjust based on the size of your GPU memory
    num_batches = len(data) // BATCH_SIZE + (1 if len(data) % BATCH_SIZE else 0)

    all_embeddings = []
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = (i + 1) * BATCH_SIZE
        # Extract data from the selected column
        batch_data = [getattr(doc, column_to_use) for doc in data[start_idx:end_idx]]
        
        batch_embeddings = generate_embeddings_for_batch(batch_data, embeddings_model)
        all_embeddings.append(batch_embeddings)

    # Combine embeddings from all batches
    all_embeddings = np.concatenate(all_embeddings, axis=0)

    db = FAISS(embeddings=all_embeddings)
    db.save_local(DB_FAISS_PATH)
    st.success("Embeddings generated successfully!")
    
    llm = load_llm()
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about the selected column in " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    # Container for the chat history
    response_container = st.container()
    # Container for the user's text input
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

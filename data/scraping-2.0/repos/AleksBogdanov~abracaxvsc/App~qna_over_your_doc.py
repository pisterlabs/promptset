import streamlit as st
from dotenv import load_dotenv
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings  # Wrapper of Embeddings in OpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message

from utils import pdf_to_string

load_dotenv()  # Loading the API keys safely

st.set_page_config(page_title="Задай въпрос")
st.markdown(
    "<h1 style='text-align: center; color: black;'>Задайте въпрос към Ваш документ </h1>",
    unsafe_allow_html=True)
hide_button_style = """
<style>
  .css-14xtw13 {    
    display: none;
  }
</style>

"""
st.markdown(hide_button_style, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Uploading the BG Constitution pdf
uploaded_file = st.file_uploader("Изберете файл", type="pdf")  # Uploading the file
if uploaded_file is not None:  # Check whether file is uploaded
    text = pdf_to_string(uploaded_file)

    # Initialize how to split the these section, using Langchain
    # Chunk overlap means, implies that if is set to 200, the chunk will contain
    # 250 tokens from the previous and 250 from the next section. The idea is not
    # to lose context.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len
    )
    # Creating chunks
    chunks = text_splitter.split_text(text)
    # Initialize Embeddings -> Converting the text into numbers
    embeddings = OpenAIEmbeddings()
    # Creating knowledge base. Using facebook FAISS Algorithm. More info in the docs.
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    message("Ask a question about this document ", is_user=False)

    with st.sidebar:
        # Get the user question
        user_input = st.text_input("Ask your question: ")
        # Prompt template is created to reduce hallucination, however it does not work well in bulgarian language
        prompt_template = f"""Question: {user_input}. Try to answer as precise and polite as you can.
        If the answer is not present in the given text, say that you do not know the answer. Be as
        detailed and descriptive as you can."""
        if user_input:
            st.session_state.messages.append(user_input)
            # Search for similarity in the knowledge base
            try:
                docs = knowledge_base.similarity_search(user_input)
                llm = OpenAI(model_name='text-davinci-003', temperature=0.3)
                chain = load_qa_chain(llm, chain_type="stuff")
                with st.spinner('Thinking ...'):
                    response = chain.run(input_documents=docs, question=prompt_template)
                st.session_state.messages.append(response)
            except:
                pass
    # Get messages from session
    messages = st.session_state.get('messages', [])
    # Display messages
    for i, msg in enumerate(messages[0:]):
        if i % 2 == 0:
            message(msg, is_user=True, key=str(i) + '_user')
        else:
            message(msg, is_user=False, key=str(i) + '_ai')

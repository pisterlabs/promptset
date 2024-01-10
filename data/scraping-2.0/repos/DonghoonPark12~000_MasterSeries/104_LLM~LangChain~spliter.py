import tempfile

import streamlit as st

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings, HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma

from PIL import Image

#----------------------------------------------------------------------------------------------------------#
# GUI
st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

header_text = 'RAG in 한국어 <span style="color: blue; font-family: Cormorant Garamond; font-size: 40px;">| Watsonx</span>'
st.markdown(f'<h1 style="color: black;">{header_text}</h1>', unsafe_allow_html=True)

# Define model and chain type options
with st.sidebar: # 사이드 바에 쌓이는 기능 들
    '''

    '''
    image = Image.open('watsonxai.jpg') 
    st.image(image, caption='watsonx.ai, a next generation enterprise studio for AI builders to train, validate, tune and deploy AI models')

    st.write("Configure model and parameters:")

    model_option = st.selectbox("Model Selected:", ["llama2-70b", "flan-ul2", "granite-13b"])
    chain_option = st.selectbox("Chain Type:", ["stuff", "refine", "mapReduce", "custom"])
    decoding_option = st.selectbox("Decoding Parameter:", ["greedy", "sample"])
    max_new_tokens = st.number_input("Max Tokens:", 1, 1024, value=256)
    min_new_tokens = st.number_input("Min Tokens:", 0, value=8)
        
    st.markdown('''
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) [LLM model](https://python.langchain.com/docs/get_started/introduction)
    ''')
    # st.markdown('Powered by <span style="color: darkblue;">watsonx.ai</span>', unsafe_allow_html=True)

st.markdown('<div style="text-align: right;">Powered by <span style="color: darkblue;">watsonx.ai</span></div>', unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------#
# LLM
def hugfacelib(repo_id):

    from langchain.embeddings import HuggingFaceHubEmbeddings

    repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding = HuggingFaceHubEmbeddings(
        task="feature-extraction",
        repo_id = repo_id,
        huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN
        )

    return embedding

def read_push_embeddings(docs):
    # repo_id="sentence-transformers/all-MiniLM-L6-v2"
    repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding = hugfacelib(repo_id)

    # vectorstore = Chroma.from_documents(
    #     documents=docs,
    #     embedding=embedding
    #     )
    
    vectorstore = FAISS.from_documents(
        documents=docs,
        embedding=embedding,
        )
        
    return vectorstore
#----------------------------------------------------------------------------------------------------------#

uploaded_files = 'D:/203_GenAI_IBM/Manual/[국문 G90 2023] genesis-g90-manual-kor-230601.pdf'

@st.cache_data
def read_pdf(uploaded_files, chunk_size=250, chunk_overlap=20):
    translated_docs = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name

            with st.spinner('Waiting for the file to upload'):
                loader = PyPDFLoader(filepath)
                data = loader.load()

                for doc in data:
                    # Extract the content of the document
                    content = doc.page_content

                    # Translate the content
                    translated_content = content # translate_large_text(content, translate_to_kor, False)

                    # Replace original content with translated content
                    doc.page_content = translated_content
                    translated_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(translated_docs)
    
    return docs

uploaded_file = st.file_uploader("Choose a PDF file", accept_multiple_files=True, type=["pdf"])

docs = read_pdf(uploaded_file)

if docs:
    db = read_push_embeddings(docs)
    st.write("\n")
else:
    st.error("문서를 업로드하십시오.")

#----------------------------------------------------------------------------------------------------------#
# GUI
st.markdown('<hr style="border: 1px solid #f0f2f6;">', unsafe_allow_html=True)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "한국 어시스턴트 입니다!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if user_question := st.chat_input("Send a message...", key="prompt"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    print(f"\n{user_question}\n")
    translated_user_input = user_question #translate_to_bahasa(user_question, False)
    print(f"{translated_user_input}\n")

if st.session_state.messages[-1]["role"] != "assistant":

    with st.chat_message("assistant"):
        with st.spinner("잠시만 기다려 주세요..."):

            try:
                db = db

                model_llm = LangChainInterface(model=model_id, credentials=creds, params=params, project_id=project_id)

                if chain_types == "custom":
                    chain = setup_qa_chain(model_llm, db, system_prompt)
                    res = chain(translated_user_input, return_only_outputs=True)
                    response = res['answer']

                else:
                    docs_search = db.similarity_search(translated_user_input, k=3)           
                    print(f"{docs_search}\n")

                    chain = load_qa_chain(model_llm, chain_type=chain_types)
                    response = chain.run(input_documents=docs_search, question=translated_user_input)

                if "<|endoftext|>" in response:
                    response = response.replace("<|endoftext|>", "")

                response = response # translate_to_kor(response, True)
                print(f"{response}\n")

            except NameError:
                response = "먼저 문서를 업로드하십시오."

            placeholder = st.empty()
            full_response = ''

            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
#----------------------------------------------------------------------------------------------------------#
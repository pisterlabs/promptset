# Bring in deps
import streamlit as st 
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

MODEL_PATH = "./models/llama-7b.ggmlv3.q4_K_M.bin"

# Customize the layout
st.set_page_config(page_title="DocQA", page_icon="🤖", layout="wide", )     
st.markdown(f"""
            <style>
            .stApp {{background-image: url("https://images.unsplash.com/photo-1468779036391-52341f60b55d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1968&q=80"); 
                     background-attachment: fixed;
                     background-size: cover}}
         </style>
         """, unsafe_allow_html=True)

# function for writing uploaded file in temp
def write_text_file(content, file_path):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False

# set prompt template
prompt_template = """You are a chat bot who loves to help people! Given the following context sections, answer the question using only the given context. If you are unsure and the answer is not explicitly writting in the documentation, say "Sorry, I don't know how to help with that."

Context sections:
{context}

Question:
{question}

Answer:"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# initialize hte LLM & Embeddings
llm = LlamaCpp(model_path=MODEL_PATH)
embeddings = LlamaCppEmbeddings(model_path=MODEL_PATH)
llm_chain = LLMChain(llm=llm, prompt=prompt)

st.title("📄 Document Question Answering")
st.text("Powered by Llama")
uploaded_file = st.file_uploader("Upload an article", type="txt")
flag = 0
if uploaded_file is not None:
    content = uploaded_file.read().decode('utf-8')
    # st.write(content)
    file_path = "temp/file.txt"
    write_text_file(content, file_path)   
    
    loader = TextLoader(file_path)
    docs = loader.load()    
    text_splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    db = Chroma.from_documents(texts, embeddings)    
    st.success("File Loaded Successfully!!")
    flag = 1
    
    # Query through LLM    
if flag == 1:
    question = st.text_input("Ask something from the file", placeholder="Find something similar to: ....this.... in the text?", disabled=not uploaded_file,)    
    if question:
        similar_doc = db.similarity_search(question, k=1)
        context = similar_doc[0].page_content
        query_llm = LLMChain(llm=llm, prompt=prompt)
        response = query_llm.run({"context": context, "question": question})        
        st.write(response)
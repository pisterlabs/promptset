"""
Features: Use Pinecone to store and retrieve documents
            Use OpenAI to generate text
            Use Streamlit to build a web app
            Use PyPDF2 to read pdf
            Use langchain to split text into chunks
            Use langchain to embed text
"""

import os
# from apikey import apikey
#import nltk
#nltk.download()

import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
#from langchain.utilities import WikipediaAPIWrapper 
from PyPDF2 import PdfReader
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.router import MultiPromptChain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma, Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_API_ENV")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")

#app framework
st.set_page_config(page_title='Algorithm Education Bot', page_icon='🧑‍💻', layout='centered')
st.title('📚 Algorithm Education Bot')
st.subheader('🤖 Welcome to your Algorithm friend!')
st.text('I have read the book "Introduction to Algorithms" by THOMAS H. CORMEN, CHARLES E. LEISERSON, RONALD L. RIVEST, and CLIFFORD STEIN.\nYou can ask me any question about algorithms in this book, and I will try my best to answer your question or give you a quiz about algorithms. \nI am still learning, so please be patient with me. 😄')

# #store pdf
# loader = UnstructuredPDFLoader("data/Introduction_to_algorithms-3rd Edition.pdf")
# data = loader.load()

# #split into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     #separator = '\n',
#     chunk_size = 2000,  #size of each chunk
#     chunk_overlap = 300, #
#     #length_function = len
# )
# texts = text_splitter.split_documents(data)

model_name = 'gpt-3.5-turbo'
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)

# initialize pinecone
pinecone.init(
    api_key=pinecone_api_key,  # find at app.pinecone.io
    environment=pinecone_env # next to api key in console
)
index_name = pinecone_index 
namespace = pinecone_namespace
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
index = pinecone.Index(pinecone_index)
#knowledge_base
# docsearch = Pinecone.from_texts(
#   [t.page_content for t in texts], embeddings, index_name=index_name)#, namespace=namespace)

@st.cache_resource
def load_pinecone_existing_index():
    #pass
    docsearch = Pinecone.from_existing_index(index_name, embedding=embeddings)
    return docsearch
docsearch=load_pinecone_existing_index()

prompt = st.text_input('Ask a question about Algorithm:')

# Prompt templates
# intro_template = PromptTemplate(
#     input_variables = ['topic'],
#     template = 'Act as a professor and write a technical, detailed introduction about background, industry usage, run time complexity, conclusion in paragraphs: {topic}',                        
# )
# runtime_template = PromptTemplate(
#     input_variables = ['topic'],
#     template = 'Act as a professor and write a technical, detailed introduction about runtime complexity of this algorithm: {topic}',
# )
# quiz_template = PromptTemplate(
#     input_variables = ['topic'],
#     template = 'Act as a professor and give a quiz about this algorithm: {topic}',
# )

intro_template = """Act as a professor, write a technical, detailed introduction about 
                    background, industry usage, run time complexity, conclusion in paragraphs: {input}
                    Wrap up response when reach 300 tokens"""
runtime_template = """Act as a professor, write a technical, short explanation about 
                    worst case of runtime complexity of all operations of this algorithm: {input}
                    Wrap up response when reach 300 tokens"""
quiz_template = """Act as a professor, give 5 quizzes about this algorithm: {input}
                    Wrap up response when reach 300 tokens"""

prompt_infos = [
    {
        "name": "intro", 
        "description": "Good for giving introduction", 
        "prompt_template": intro_template,
        "input_variables": ["input"],
    },
    {
        "name": "runtime", 
        "description": "Good for answering runtime complexity", 
        "prompt_template": runtime_template,
        "input_variables": ["input"],
    },
    {
        "name": "quiz", 
        "description": "Good for giving quiz or question", 
        "prompt_template": quiz_template,
        "input_variables": ["input"],
    },
]

#Llms
llm = ChatOpenAI(model_name= model_name, temperature = 0.1, max_tokens = 400)#, top_p = 0.2, frequency_penalty = 0.8, presence_penalty = 0.1)
#intro_chain = LLMChain(llm = llm, prompt = intro_template, output_key = 'introduction')
#code_example_chain = LLMChain(llm = llm, prompt = code_example_template, output_key = 'code_example')
#sequential_chain = SequentialChain(chains = [intro_chain, code_example_chain], input_variables = ['topic'], 
#                               ouput_variables = ['intro', 'code'], verbose = True)
#wiki = WikipediaAPIWrapper()

#show stuff to screen if promted
if prompt:
    topic = prompt
    input = topic
    #docs = docsearch.similarity_search(prompt)
    #chain = load_qa_chain(llm, chain_type = 'stuff') #prompt = intro_template, 
    chain = MultiPromptChain.from_prompts(llm, prompt_infos, verbose=True)
    with get_openai_callback() as cb:
        response = chain.run(input = prompt, retriever = docsearch.as_retriever(),)#, input_documents = docs, )
        print(cb)

    #st.write("**Introduction:** :sunglasses:")
    st.write(response)
    #check why it can work on unrelated information
 
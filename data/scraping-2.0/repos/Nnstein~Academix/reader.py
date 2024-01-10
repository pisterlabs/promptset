import os
from  langchain.llms import OpenAI
import streamlit as st
from api import apikey
# from config import settings
# import dotenv``



#libraries for readerpy
from langchain.embeddings import OpenAIEmbeddings
from  langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import(
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#libraries for app.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory


# dotenv.load_dotenv()
# os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = apikey
# os.environ['SERPAPI_API_KEY'] = serpapikey
# apikey = settings.apikey

# llms
llm = OpenAI(temperature = 0.9)
embeddings = OpenAIEmbeddings()


memory = ConversationBufferMemory(memory_key="chat_history")

# create and dload the pdf loeader
loader = PyPDFLoader('CSC HANDBOOK FOR NUC.pdf')

# split pdf pages
pages = loader.load_and_split()

# load the split pages into chromadb
store = Chroma.from_documents(pages,embeddings,collection_name='csc_handbook')

# create vectorstore info object - meta repo
vectorstore_info = VectorStoreInfo(
    name = 'Csc department guide',
    description = 'A guide for students of Csc Dept trying to navigaet their way on campus',
    vectorstore=store
)

# convert the document store into a langchain toolkit likr the wikipedia toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# add the toolkit to an end-to-end large Composer
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose = True
)

# create a text input prompt for the user
st.title('🔛 Academix Assist')
prompt = st.text_input('Plug in your prompt here')

if prompt:
    response  = agent_executor.run(prompt)
    search= store.similarity_search_with_score(prompt)
    # st.write(search[0][0].page_content)
    st.write(response)
    

    # with streamlit expander, 
    # with st.expander('Document Similarity Search'):
    #     #find the relevant pages
    #     search= store.similarity_search_with_score(prompt)
    #     st.write(search[0][0].page_content)
    



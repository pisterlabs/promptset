from langchain.document_loaders import GitLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language,CharacterTextSplitter
from langchain.vectorstores import DeepLake, VectorStore
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
# from langchain.retrievers import BaseRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader,DirectoryLoader
from langchain.schema import AIMessage, HumanMessage,SystemMessage,Document
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import sys
sys.path.insert(0, './src')
import os
import re 
import streamlit as st
import logging
import shutil
import time
import deeplake
import subprocess
from dotenv import load_dotenv
from typing import List
from constants import *

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['ACTIVELOOP_TOKEN'] = os.getenv('ACTIVELOOP_TOKEN')
logger = logging.getLogger(APP_NAME)


    

# data_source = input("Enter the URL of the repository you want to analyze: ")

   

def handle_load_error(e: str = None) -> None:
    e = "Invalid GitHub URL. Please !"
    error_msg = f"Failed to load with Error:\n{e}"
    # st.error(error_msg, icon="warning")
    logger.info(error_msg)
    # st.stop()



def load_git(data_source: str, chunk_size: int = 1000) -> List[Document]:
    # We need to try both common main branches
    # Thank you github for the "master" to "main" switch
    repo_name = data_source.split("/")[-1].split(".")[0]
    repo_path = str(DATA_PATH / repo_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
    )
    branches = ["main", "master"]
    for branch in branches:
        if os.path.exists(repo_path):
            data_source = None
        try:
            docs = GitLoader(repo_path, data_source, branch,file_filter = lambda file_path: file_path.endswith((".py", ".html", ".c", ".js", ".md", ".tsx", ".jsx" , ".txt"))).load_and_split(
                text_splitter
            )
            break
        except Exception as e:
            logger.info(f"Error loading git: {e}")
    if os.path.exists(repo_path):
        # cleanup repo afterwards
        # shutil.rmtree(repo_path)
        os.system('rmdir /S /Q "{}"'.format(repo_path))
    try:
        return docs
    except Exception as e:
        handle_load_error()
        
        
@st.cache_data
def clean_data_source_string(data_source_string: str) -> str:
    # replace all non-word characters with dashes
    # to get a string that can be used to create a new dataset
    dashed_string = re.sub(r"\W+", "-", data_source_string)
    cleaned_string = re.sub(r"--+", "- ", dashed_string).strip("-")
    return cleaned_string


username = "abhishekrp2002"  # replace with your username from app.activeloop.ai
def setup_vector_store(data_source: str, chunk_size: int = CHUNK_SIZE) -> VectorStore:
    # either load existing vector store or upload a new one to the hub
    embeddings = OpenAIEmbeddings(
        disallowed_special=(), openai_api_key=os.environ['OPENAI_API_KEY']
    )
    data_source_name = clean_data_source_string(data_source)
    dataset_path = f"hub://{username}/{data_source_name}-{chunk_size}"
    if deeplake.exists(dataset_path, token=os.environ['ACTIVELOOP_TOKEN']):
        with st.spinner("Loading vector store..."):
            logger.info(f"Dataset '{dataset_path}' exists -> loading")
            vector_store = DeepLake(
                dataset_path=dataset_path,
                read_only=True,
                embedding_function=embeddings,
                token=os.environ['ACTIVELOOP_TOKEN'],
            )
    else:
        # with st.spinner("Reading, embedding and uploading data to hub..."):
            logger.info(f"Data '{dataset_path}' does not exist -> uploading")
            docs = load_git(data_source, chunk_size)
            vector_store = DeepLake.from_documents(
                docs,
                embeddings,
                dataset_path=dataset_path,
                token=os.environ['ACTIVELOOP_TOKEN'],
            )
    return vector_store


    

@st.cache_data
def analysis_main(data_source_name:str,data_source: str, chunk_size: int = CHUNK_SIZE):
 # create the langchain that will be called to generate responses
    vector_store = setup_vector_store(data_source, CHUNK_SIZE)
    retriever = vector_store.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["k"] = 5

    repository_name = data_source_name
    repository_url = data_source

# Step 6: Create a ConversationalRetrievalChain using ChatOpenAI and the retriever


# Step 7: Define a list of questions to ask about the codebase

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False)



    query = """

You are Codebase AI. You are a superintelligent AI that answers questions about codebases and can understand the technical complexity if the codebase.

    You are:
    - helpful & friendly
    - good at answering complex questions in simple language
    - an expert in all programming languages
    - able to infer the intent of the user's question

    The user will ask a question about their codebase, and you will answer it.

Remember that you are intelligent AI system that can directly analyze the codebase of the repository and create a scoreing model internally. 

If the codebase if fairly simple , then It is technically less complex and if the codebase is complex then it is technically more complex.

The technical complexity of a codebase or GitHub project can be determined by various factors. Some of these factors are:

1. Size of the codebase: Larger codebases generally tend to be more complex due to the higher number of components and interactions among them.

2. Modularity and Structure: A well-structured and modular codebase with clear separation of concerns can be easier to understand and maintain, whereas a monolithic codebase with mixed responsibilities can increase complexity.

3. Technologies and libraries used: The choice of technologies, programming languages, and libraries can affect the complexity of a project. Some technologies have steeper learning curves and specializations, which can make a project more complex.

4. Dependencies: The number of dependencies and their complexity can also impact the overall complexity of a project.

5. Code quality: Well-written code with proper documentation, comments, and adherence to coding standards or best practices tends to be easier to understand and maintain, which reduces complexity.

6. Testing: The presence and quality of automated tests can affect the complexity of a project. Comprehensive tests can make it easier to understand the intended functionality of the code and improve maintainability.

To calculate a technical complexity score for a project, you can use a weighted combination of these factors. Here's an example approach:

1. Assign a score for each factor on a scale of 1 to 5, with 1 being the least complex and 5 being the most complex.

2. Assign weights to each factor based on their importance (for example, a weight of 0.3 for code quality, 0.2 for dependencies, etc.). Ensure the sum of all weights equals 1.

3. Calculate the weighted average of the scores for each factor using their assigned weights.

```
technical_complexity_score = (score1 * weight1) + (score2 * weight2) + ... + (scoreN * weightN)
```

Display the answer in the following format - Example :  

Technical Complexity Score of the Repository is 4.5/5 . 
Analysis -  ______ .


Use the above metrics and factors to calculate the technical complexity score of the codebase.


Now answer the following questions about the codebase:

What is the main functionality of this codebase? What is the Technical Complexity Score of the codebase?Find the technical complexity score of the current project and Provide a detailed analysis for that given score , basedd on the factors and your own reasoning.

If you do not have enough information to predict the complexity score, assign a smaller value and state your reasons for doing so.

Let's think step by step about how to answer this question:
"""


# Step 8: Use the RetrievalChain to generate context-aware answers

    result = qa({"query": query})
    print(result['result'])
    return (repository_name,repository_url,result['result'])


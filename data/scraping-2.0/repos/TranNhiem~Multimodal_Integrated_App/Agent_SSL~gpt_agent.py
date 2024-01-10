'''
TranNhiem 2023/05/08
Building the Agent for Information Retrieval and Question Answering Information. 
    1/ The first implementation work only with Text paragraph objective:
    2/ Working with Images and Video Objectives 
    3/ Working with Audio Objectives 

1/ The first implementation work only with Text paragraph objective:
    + Implement the Agent for Information Retrieval and Question Answering Information.
    (1) The Agent will be able to retrieve the information from the given conversation script from (Youtube Video, Future work via Direct Upload Video )
        + Give you Summary main idea of the conversation --> Answer Question your question
    (2) The Agent will be able to answer the question from the given paragraph
    (3) The Agent will be able to answer the question from the given textbook and lecture notes
    (4) The Agent will be able to answer the question from the given Small Language tasks dataset


'''


#---------------Import and Install Library------------------------------

import os
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
## Import the Embedding Model
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings # OpenAIEmbeddings
## VectorDatabase 

from langchain.vectorstores import FAISS 
import pinecone 
from langchain.vectorstores import Pinecone

## Loading LLM Model
from langchain.llms import OpenAI
# Import Azure OpenAI

# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2022-12-01"
# os.environ["OPENAI_API_BASE"] = "..."
# os.environ["OPENAI_API_KEY"] = "..."
from langchain.llms import AzureOpenAI

from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
import textwrap


load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    chain= LLMChain(llm, prompt)
    response= chain.run(question=query, docs=docs_page_content)
    response= response.replace("\n", "")
    return response, docs 


# Example usage:
video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
db = create_db_from_youtube_video_url(video_url)

query = "What are they saying about Microsoft?"
response, docs = get_response_from_query(db, query)
print(textwrap.fill(response, width=85))

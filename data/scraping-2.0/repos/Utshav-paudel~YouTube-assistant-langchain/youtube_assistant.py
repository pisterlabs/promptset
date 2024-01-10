#@ Creating a youtube assistant that will help you convert a youtube video to script

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

url="https://youtu.be/BoutTY8XHSc?si=RFqU6VHQiFBENdop"                                            # link of video
embeddings = OpenAIEmbeddings()

def yotube_url_to_vector_db(url:str) -> FAISS:                            
    loader = YoutubeLoader.from_youtube_url(youtube_url=url)                                       # uses langchain component to load yotube url
    transcripts = loader.load()                                                                    # create transcript of video using yotube loader
    splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap =50)                 # split the trancript
    docs = splitter.split_documents(transcripts)
    # vector databse to store the embeddings
    db = FAISS.from_documents(docs, embeddings)                                                   # store the embedding into vector db of docs
    return db

def get_response_from_query(db, query, k=4):           # k determines number of chunks
    "we will use text-davinci-003 which has cap of 4096 tokens k determine number of 1000 chunk"
    docs = db.similarity_search(query, k=k)
    docs_content =" ".join([d.page_content for d in docs])
    llm = OpenAI(model_name = "text-davinci-003")                                
    template = PromptTemplate(input_variables=['question','docs'],template=
                   """You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.""")
    chain = LLMChain(prompt = template, llm=llm)
    response = chain.run(question=query,docs= docs_content)
    return response
db = yotube_url_to_vector_db(url)
query = "What are the tools to hack your brain ?"
print(get_response_from_query(db,query))
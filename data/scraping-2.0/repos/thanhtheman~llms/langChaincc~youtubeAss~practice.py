# load the youtube video into transcript
# split the transcript into chunks
# put the chunks into the vector db
# retrieve the relevatn documents and feed it into an llm to generate responses
#youtube url https://www.youtube.com/watch?v=2eWuYf-aZE4&t=2s

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()

def create_vector_db(youtube_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(youtube_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    print("create_vector_db docs type is " + str(type(docs)))
    print(docs)
    db = FAISS.from_documents(docs, embeddings)
    return db

def create_response(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    print(docs)
    document = " ".join([d.page_content for d in docs])
    llm= ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    prompt = PromptTemplate(template ="""
    You are a helpful Youtube assistant that can answer questions about
    videos bsaed on the video's transcript. 
    Answer the following questions: {question}
    By searching the following video transcript: {document} 
    Only use the factual information from the transcript to answer the question.
    If you feel like you don't have enough information to answer the question. Please say 
    "I don't know".
    Your answers should be detailed.
    """,
    input_variables=["question", "document"])
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.predict(question=query, document=document).replace("/n","")
    print(response)

db = create_vector_db("https://www.youtube.com/watch?v=2eWuYf-aZE4&t=2s")
create_response(db, "what is the key point?", k=4)    









from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

video_url = os.getenv('VIDEO_URL', "https://www.youtube.com/watch?v=pMFv6liWK4M")
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = create_vector_db_from_youtube_url(video_url)

def get_response_from_query(query, k=4):
    # text-danvinci can handle 4097 tokens

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript and/or by using your general knowledge. Fallback to general knowledge only
        when you the video transcript has not even slightly relevant context. But, mention that "the video doesn't specify anything relevant to
        the question" in the video transcript in cases where you fallback as a prefix.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs} and/or using your general knowledge. Fallback to general knowledge only
        when you the video transcript has not even slightly relevant context. But, mention that "the video doesn't specify anything relevant to
        the question" in the video transcript in cases where you fallback as a prefix.
                
        Then, If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response

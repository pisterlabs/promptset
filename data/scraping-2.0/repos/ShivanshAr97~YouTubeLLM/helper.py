from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings()

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = GooglePalm(temperature=0.8)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You  answer questions about youtube videos provided to you based on the video's transcript.
        Answer the question: {question}
        By searching the the video transcript provided: {docs}
        Use only the real facts and information from the transcript to answer the question.
        If you feel like you don't get the answer for the question then simply say "I don't know".        
        Your answers should be verbose explaining the answer in a single paragraph of 70 to 100 words unless not specified.
        """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs
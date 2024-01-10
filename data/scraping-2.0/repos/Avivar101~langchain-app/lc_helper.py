from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch

from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()

def ceate_vector_db_from_youtube(video_url: str) -> DocArrayInMemorySearch:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, k=4):
    query = "Return Detailed Summary"
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="text-davinci-003")
    prompt = PromptTemplate(
        input_variables=["docs"],
        template="""
        You are a helpful youtube assistant that can give a detailed summary and highlight key points of youtube videos based on a youtube's transcript.

        By searching the following transcript: {docs}

        Only use the factal information from the transcripts to give replies.

        Your replies should be detailed.
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response
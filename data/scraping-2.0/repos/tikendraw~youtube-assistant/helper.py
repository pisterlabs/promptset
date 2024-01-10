import logging
from functools import cache

from dotenv import load_dotenv
from icecream import ic
from langchain.chains import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='youtube-assistant.log')
logger = logging.getLogger(__name__)

load_dotenv()

model = 'mistral'
embeddings = OllamaEmbeddings(model=model)
llm = Ollama(
    model=model,
    temperature=.2,
    repeat_penalty=1.3
)

@cache
def get_youtube_docs(url):
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()

    return docs

def get_youtube_video_to_db(youtube_url, embeddings, chunk_size=1000, chunk_overlap=100):
    docs = get_youtube_docs(youtube_url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    db = FAISS.from_documents(texts, embeddings)
    return db

def get_answer(query, llm, db, k=3):
    logger.info('Reading Docs...')
    try:
        docs = db.similarity_search(query, k=k)
        docs_content = '\n'.join([p.page_content for p in docs])
        prompt = PromptTemplate(
            input_variables=['query'],
            template='''
            You are a helpful assistant.
            Answer the following question based on the provided context, or say "Hmm, I don't know." if you don't know the answer.
            Question: {query}
            Context: {context}
            Answer:
            '''
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        logger.info('Getting Answers...')
        return chain.run(query=query, context=docs_content)
    except Exception as e:
        # Log errors in detail
        logger.error(f'Error occurred: {e}', exc_info=True)
        return None


if __name__ == '__main__':
    youtube_url = "https://www.youtube.com/watch?v=U_M_vDChJQ"
    
    try:
        db = get_youtube_video_to_db(video_id, embeddings)
    except Exception as e:
        raise e
    
    query = 'who is blackmailing elon? '
    result = get_answer(query, llm, db, k = 5)
    logger.info(f'Result for query "{query}": {result}')
    
    print(query)
    print(result)
    print()
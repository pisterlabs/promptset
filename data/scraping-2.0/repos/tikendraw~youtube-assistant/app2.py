import logging
import time

import streamlit as st
from icecream import ic
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from main import extract_youtube_video_id, get_answer, get_youtube_video_to_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = 'mistral'

@st.cache_resource
def get_model_and_embedding(model_name):
    embeddings = OllamaEmbeddings(model=model_name)
    llm = Ollama(
        model=model,
        temperature=.2,
        repeat_penalty=1.3,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm, embeddings

def main():
    st.title(':robot: Youtube-assistant') 
    st.header("Youtube Assistant")
    st.text("""Video is too long? Zoned out? Ask it here!""")

    youtube_url = st.text_input('Youtube url here', placeholder='https://www.youtube.com/watch?v=XXXXXXXXXXX')
    query = st.text_input('Your curiosity...', placeholder='What is that...') 
    
    submit = st.button('Answer Me.')
    llm, embeddings = get_model_and_embedding(model_name=model)

    if youtube_url and query and submit:
        # num = ''
        # with st.empty():
        #     for i in range(1000):
        #         num = num+' '+str(i)
        #         st.write(num)
        #         time.sleep(0.07)
        
        try:
            video_id = extract_youtube_video_id(youtube_url)
        except ValueError as e:
            st.error(e)
            
        try:
            db = get_youtube_video_to_db(video_id, embeddings)
        except Exception as e:
            st.text(e)
            return    
        
        # st.markdown(f"# {query}")
        write_effect(query)
        # Create an empty element to update dynamically
        # output = ''
        # with st.empty():
        #     for token in get_answer_generator(query, llm, db, k=5):
        #         output += token
        #         st.text(output)
        #         time.sleep(0.005)  # Introduce a small delay for a smoother display
        #         print(token , end=' ')
                
        write_effect(get_answer_generator(query=query, llm=llm, db=db, k=3), delay=0.05)
        # st.write(output)
        logger.info(f'Result for query "{query}": Completed')


def write_effect(genetator, delay=0.01):
    output = ''
    with st.empty():
        for i in genetator:
            output += i
            if len(output)%50 > 0 and i==' ':
                output += '\n'
                
            st.write(output)
            # st.code(output, language=None, line_numbers=False)

            time.sleep(delay)  # Introduce a small delay for a smoother display
            print(i , end=' ')
            
def get_answer_generator(query, llm, db, k=3):
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
        print('Trying...')
        # Yield each token as it is generated
        return chain.run(query=query, context=docs_content)

    except Exception as e:
        # Log errors in detail
        logger.error(f'Error occurred: {e}', exc_info=True)
        return None

if __name__=='__main__':
    main()

from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import streamlit as st 
import time
import re
#from apikey import apikey 
#os.environ['OPENAI_API_KEY'] = apikey
#OPENAI_API_KEY = 'Your key'

def is_valid_youtube_url(url):
    """
    Function to check if a URL is a valid YouTube URL
    """
    match = re.search(r"(http(s)?:\/\/)?((w){3}.)?youtu(be|.be)?(\.com)?\/.+", url)
    return bool(match)

st.title('✍️🔗 YouTube Links Summary ✍️')

st.markdown("<h2 style='color:blue;'>Enter Your openai API KEY</h2>", unsafe_allow_html=True)
key= st.text_input('Ex: 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') 

st.markdown("<h1>Enter Your YouTube Link </h1>", unsafe_allow_html=True)
link = st.text_input('') 

if link and is_valid_youtube_url(link):
    loader = YoutubeLoader.from_youtube_url(link, add_video_info=True)
    result = loader.load()
    llm = OpenAI(temperature=0, openai_api_key=key)
    chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
    if key: 
        res= chain.run(result)
        with st.spinner('Loading...'):
            st.write(res)
            time.sleep(2)
            st.success('Done!')
else:
    st.error("The provided URL is not a valid YouTube URL.")

from langchain.chains.openai_functions.openapi import get_openapi_chain
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs, podcast_docs, tmdb_docs
import os
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler


def get_unique_file_path(base_path, file_name_pattern):
    index = 1
    file_path = os.path.join(base_path, file_name_pattern.format(index))
    
    while os.path.exists(file_path):
        index += 1
        file_path = os.path.join(base_path, file_name_pattern.format(index))
        
    return file_path



load_dotenv()


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")
listen_api_key = os.environ["LISTEN_API_KEY"]



headers = {"Authorization": f"Bearer {os.environ['TMDB_BEARER_TOKEN']}"}
chain = APIChain.from_llm_and_api_docs(llm,tmdb_docs.TMDB_DOCS, headers=headers, verbose=True,) 

podcast_headers = {"X-ListenAPI-Key": listen_api_key}

podcast_chain = APIChain.from_llm_and_api_docs(llm, podcast_docs.PODCAST_DOCS, headers=podcast_headers, verbose=True)

st.title("miniAGI :computer:")
st.subheader("OpenAI API chain")
api_choice = st.selectbox("Choose the API:", ("TMDB", "Podcast"))

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    # Determine which API chain to run based on the user's selection
    selected_chain = chain if api_choice == "TMDB" else podcast_chain
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        # Define the base path and file name pattern
        base_path = "documents/api_yaml"
        file_name_pattern = "api_chain_{}.yaml"

        # Get the unique file path
        unique_file_path = get_unique_file_path(base_path, file_name_pattern)

        # Save the chain to the unique file path
        chain.save(file_path=unique_file_path)

        podcast_response = podcast_chain.run(prompt, callbacks=[st_callback])

        response = selected_chain.run(prompt, callbacks=[st_callback])
         

        
        st.info(f"Saved chain to documents/api_yaml/{unique_file_path}")

        st.write(response)
        
        
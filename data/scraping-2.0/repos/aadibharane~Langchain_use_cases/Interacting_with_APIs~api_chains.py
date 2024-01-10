#API Chains
#This notebook showcases using LLMs to interact with APIs to retrieve relevant information.

from langchain.chains.api.prompt import API_RESPONSE_PROMPT

from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate


import os
os.environ["OPENAI_API_KEY"] ="OPENAI_API_KEY"
import os
os.environ['TMDB_BEARER_TOKEN'] = "TMDB_BEARER_TOKEN"
from langchain.chains.api import tmdb_docs
from langchain.chains.api import open_meteo_docs


from langchain.llms import OpenAI

def api_chain():
    llm = OpenAI(temperature=0)

    #OpenMeteo Example
    chain_new = APIChain.from_llm_and_api_docs(llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=True)
    chain_new.run('What is the weather like right now in Munich, Germany in degrees Fahrenheit?')

    headers = {"Authorization": f"Bearer {os.environ['TMDB_BEARER_TOKEN']}"}
    chain = APIChain.from_llm_and_api_docs(llm, tmdb_docs.TMDB_DOCS, headers=headers, verbose=True)
    chain.run("Search for 'Avatar'")

    #Listen API Example
    # import os
    # from langchain.llms import OpenAI
    # from langchain.chains.api import podcast_docs
    # from langchain.chains import APIChain

    # # Get api key here: https://www.listennotes.com/api/pricing/
    # listen_api_key = 'xxx'

    # llm = OpenAI(temperature=0)
    # headers = {"X-ListenAPI-Key": listen_api_key}
    # chain = APIChain.from_llm_and_api_docs(llm, podcast_docs.PODCAST_DOCS, headers=headers, verbose=True)
    # chain.run("Search for 'silicon valley bank' podcast episodes, audio length is more than 30 minutes, return only 1 results")
api_chain()
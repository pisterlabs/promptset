
from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain

import os

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

load_dotenv()
OPENAI_MODEL = "gpt-3.5-turbo"
# OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    # setup llm 
    llm = ChatOpenAI()
    chain_new = APIChain.from_llm_and_api_docs(
        llm, open_meteo_docs.OPEN_METEO_DOCS, verbose=False)
    target = input("Please enter your area of interest: ")
    query = (f"What is is the weather like right now in {target} in degrees celcius. ")
    result = chain_new.run(
        query
    )
    print(result)


if __name__ == "__main__":
    main()
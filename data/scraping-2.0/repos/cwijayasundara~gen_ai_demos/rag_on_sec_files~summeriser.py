import openai
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def summerise_pdf():
    prompt_template = """Write a concise summary of the following:
    "{text}"
    in point form. CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm = ChatOpenAI(temperature=0,
                     model_name="gpt-3.5-turbo-1106")

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain,
                                      document_variable_name="text")

    loader = UnstructuredPDFLoader("docs/2023q3-alphabet-earnings-release.pdf")
    docs = loader.load()

    return stuff_chain.run(docs)

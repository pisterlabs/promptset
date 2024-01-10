from langchain.chat_models import AzureChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import os

DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME") 
TEMPERATURE = 0

def summarize_text(text):
    """
    Summarize the given text using the langchain.

    Args:
        text (str): Text to be summarized.

    Returns:
        str: Summarized text.
    """
    llm = AzureChatOpenAI(temperature=TEMPERATURE, deployment_name=DEPLOYMENT_NAME, api_version="2023-05-15")
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    doc = Document(page_content=text)
    return chain.run([doc])
import json
from langchain.llms import BaseLLM, OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, RetrievalQA
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from mykey import key
import os
os.environ["OPENAI_API_KEY"] = key


# Set up a knowledge base
def setup_knowledge_base(product_catalog: str = None):
    """
    The product knowledge base is a JSON file.
    """
    # load product catalog
    with open(product_catalog, "r") as f:
        product_catalog = json.load(f)

    # convert the JSON data to text
    product_catalog_text = json.dumps(product_catalog)

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog_text)

    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def get_tools(product_catalog):
    # query to get_tools can be used to be embedded and relevant tools found
    # see here: https://langchain-langchain.vercel.app/docs/use_cases/agents/custom_agent_with_plugin_retrieval#tool-retriever

    # we only use one tool for now, but this is highly extensible!
    knowledge_base = setup_knowledge_base(product_catalog)
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information",
        )
    ]

    return tools

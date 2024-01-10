from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json
import os
from api.pinecone_db import get_db_instance
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import openai





################################################################################
# Make OpenAI query
################################################################################
def query_related(query):

    db = get_db_instance()
    
    prompt_template = """
    You will receive a product name delimited with <>. Your task is to use the following pieces of context to find a maximum of three products that are related to the product delimited with <>. Once found, return an array with the id of each of those products.
    # If you can not find any related products, just return an empty array.
    # Don't return duplicated products.
    # Don't return the product provided in the final result.
    # Don't return an object.
    # Return a maximum of 3 products.

    {context}

    Product: <{question}>
    Answer in JSON format:"""

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,
                       openai_api_key=os.getenv('OPENAI_API_KEY')),
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )

    res = qa.run(query)
    
    return json.loads(res)

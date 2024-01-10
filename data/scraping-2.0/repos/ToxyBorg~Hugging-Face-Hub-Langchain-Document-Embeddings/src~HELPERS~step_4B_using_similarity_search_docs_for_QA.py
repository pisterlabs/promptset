"""
    This function is used to implement a question answering system. 
    
    The function takes in a list of Document objects, a query string, and two optional parameters for the Hugging Face Hub API token and repository ID.

    The function uses the HuggingFaceHub class from the llms module to load a pre-trained language model from the Hugging Face Hub. 
    
    It then uses the load_qa_chain() function from the question_answering module to load a question answering chain. 
    
    Finally, it uses the chain to find the answer to the query.
"""

import os
from typing import List

from langchain.llms import HuggingFaceHub

from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def Q_and_A_implementation(
    similarity_search_docs: List[Document],
    query: str,
    huggingfacehub_api_token: str | None = None,
    huggingfacehub_repo_id: str = os.getenv("HUGGINGFACE_REPO_ID"),
) -> str:
    """
    Implement a question answering system.

    Args:
        - similarity_search_docs (List[Document]): A list of Document objects.
        - query (str): The query string.
        - huggingfacehub_api_token (str): The Hugging Face Hub API token.
        - huggingfacehub_repo_id (str): The repository name.

    Returns:
        - str: The answer to the query.
    """

    llm = HuggingFaceHub(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id=huggingfacehub_repo_id,
        model_kwargs={"temperature": 0.1, "max_new_tokens": 300},
    )

    # Load the question answering chain
    chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        verbose=True,
    )

    # Use the chain to find the answer to the query
    Q_and_A_answer = chain.run(
        input_documents=similarity_search_docs, question=query, raw_response=True
    )

    return Q_and_A_answer

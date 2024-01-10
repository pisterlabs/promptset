from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from typing import List, Optional
import re
from .helpers.utils import (
    instruction_response,
    output_format_checker,
    generate_full_prompt,
    retry_on_rate_limit_error,
)
from .helpers.create_logger import create_logger
import ast


load_dotenv()
import os
import time

LOG = create_logger()


@retry_on_rate_limit_error(wait_time=30)
def check_responses(bbg_docs, bloomberg_docs, code_docs, prompt_strike):
    bbg_prompt = generate_full_prompt(bbg_docs, prompt_strike)
    bloomberg_prompt = generate_full_prompt(bloomberg_docs, prompt_strike)
    code_prompt = generate_full_prompt(code_docs, prompt_strike)
    result_bbg = instruction_response(bbg_prompt)
    result_bloomberg = instruction_response(bloomberg_prompt)
    result_code = instruction_response(code_prompt)

    list_of_results = [result_bbg, result_bloomberg, result_code]

    final_check_template = PromptTemplate.from_template(
        "The list of strike prices of the underlyings is a list of floats"
        "You found the following results in the first analysis: "
        "List of strike prices: {first_result}"
        "If they are not in the format of a list of floats, respond 'retry' to rerun the analysis"
        "Now we need to make sure that the list of tickers is in the format of tickers"
        "You fund the following results for the stike prices accoridng to the underlyings: {first_result}"
        "Your response should be a list"
        "Example of your response: [float,float,...]"
    )

    final_check = final_check_template.format(first_result=list_of_results)

    final_result = instruction_response(final_check)

    pattern = r"\[[^\]]*\]"
    # Find all matches
    matches = re.findall(pattern, final_result)
    if len(matches) == 1:
        final_result = matches[0]
        final = ast.literal_eval(final_result)
        checks = 0
        for i in final:
            if isinstance(i, float):
                checks += 1

        if checks == len(final):
            LOG.info(f" Strike prices extracted:{final} ")
            return final
        else:
            return None


@output_format_checker(max_attempts=5, desired_format=list)
def get_strike(document_name: str) -> Optional[List[float]]:
    LOG.info("Loading document to extract strike prices")

    with open(document_name, "r") as f:
        loaded_json = json.load(f)

    documents = loaded_json["full_text"]
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0.2, separator=" "
    )
    texts = text_splitter.split_text(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    bloomberg = retriever.get_relevant_documents(
        "Bloomberg",
    )
    bbg = retriever.get_relevant_documents(
        "BBG",
    )
    code = retriever.get_relevant_documents(
        "Underlyings Index Code",
    )

    prompt_strike = PromptTemplate.from_template(
        "Extract the strikeprices based on the tickers of the underlyings (components/stocks) of term sheet"
        "The list of underlyings is a list of stocks that the term sheet will include in its investment portfolio"
        "You need to find the strike price of the options chosen to be the underlyings of the term sheet"
        "The strike price is the price at which the option can be exercised"
        "Make sure that your response is in teh format of a list"
        "Given the following context:"
        "Context: {context}"
        "Your response format example: [float,float,...]"
    )

    res = check_responses(bbg, bloomberg, code, prompt_strike)
    return res

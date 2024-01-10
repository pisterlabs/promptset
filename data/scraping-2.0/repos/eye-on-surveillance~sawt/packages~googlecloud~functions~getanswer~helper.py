from langchain.vectorstores.faiss import FAISS
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
import logging
import pandas as pd


logger = logging.getLogger(__name__)


"""Parse field from JSON or raise error if missing"""


def parse_field(request_json, field: str):
    if request_json and field in request_json:
        return request_json[field]
    else:
        raise ValueError(f"JSON is invalid, or missing a '${field}' property")


def get_dbs():
    dir = Path(__file__).parent.absolute()
    general_embeddings, in_depth_embeddings = create_embeddings()

    general_faiss_index_path = dir.joinpath("cache/faiss_index_general")
    in_depth_faiss_index_path = dir.joinpath("cache/faiss_index_in_depth")
    voting_roll_df_path = dir.joinpath("cache/parsed_voting_rolls.csv")

    db_general = FAISS.load_local(general_faiss_index_path, general_embeddings)
    db_in_depth = FAISS.load_local(in_depth_faiss_index_path, in_depth_embeddings)
    logger.info("Loaded databases from faiss_index_general and faiss_index_in_depth")
    voting_roll_df = pd.read_csv(voting_roll_df_path)
    return db_general, db_in_depth, voting_roll_df


def create_embeddings():
    llm = ChatOpenAI(model="gpt-4")

    base_embeddings = OpenAIEmbeddings()

    general_prompt_template = """
    As an AI assistant, your role is to provide concise, balanced summaries from the transcripts of New Orleans City Council meetings in response to the user's query "{user_query}". Your response should not exceed one paragraph in length. If the available information from the transcripts is insufficient to accurately summarize the issue, respond with 'Insufficient information available.' If the user's query extends beyond the scope of information contained in the transcripts, state 'I don't know.'
    Answer:"""

    in_depth_prompt_template = """
    As an AI assistant, use the New Orleans City Council transcript data that you were trained on to provide an in-depth and balanced response to the following query: "{user_query}" 
    Answer:"""

    general_prompt = PromptTemplate(
        input_variables=["user_query"], template=general_prompt_template
    )
    in_depth_prompt = PromptTemplate(
        input_variables=["user_query"], template=in_depth_prompt_template
    )

    llm_chain_general = LLMChain(llm=llm, prompt=general_prompt)
    llm_chain_in_depth = LLMChain(llm=llm, prompt=in_depth_prompt)

    general_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain_general,
        base_embeddings=base_embeddings,
    )
    in_depth_embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain_in_depth, base_embeddings=base_embeddings
    )

    return general_embeddings, in_depth_embeddings


def sort_retrived_documents(doc_list):
    docs = sorted(doc_list, key=lambda x: x[1], reverse=True)

    third = len(docs) // 3

    highest_third = docs[:third]
    middle_third = docs[third : 2 * third]
    lowest_third = docs[2 * third :]

    highest_third = sorted(highest_third, key=lambda x: x[1], reverse=True)
    middle_third = sorted(middle_third, key=lambda x: x[1], reverse=True)
    lowest_third = sorted(lowest_third, key=lambda x: x[1], reverse=True)

    docs = highest_third + lowest_third + middle_third
    return docs

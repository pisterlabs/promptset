import logging
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
import json
import os

from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from helper import sort_retrived_documents
from api import RESPONSE_TYPE_DEPTH, RESPONSE_TYPE_GENERAL, RESPONSE_TYPE_VARIED

logger = logging.getLogger(__name__)


def timestamp_to_seconds(timestamp):
    if 'timestamp not available' in timestamp:
        return None  # or another default value like -1 or 0
    start_time = timestamp.split("-")[0]  # Split by '-' and take the first part
    h, m, s = [int(i) for i in start_time.split(":")]
    return h * 3600 + m * 60 + s


def process_responses_llm(responses_llm, docs=None, card_type = "in_depth"):
    generated_responses = responses_llm.split("\n\n")
    responses = []
    citations = []

    if docs:
        generated_titles = [
            doc[0].metadata.get("title", doc[0].metadata.get("source", ""))
            for doc in docs
        ]
        page_numbers = [doc[0].metadata.get("page_number") for doc in docs]
        generated_sources = [
            doc[0].metadata.get("source", "source not available") for doc in docs
        ]
        publish_dates = [
            doc[0].metadata.get("publish_date", "date not available") for doc in docs
        ]
        timestamps = [
            doc[0].metadata.get("timestamp", "timestamp not available") for doc in docs
        ]
        urls = [doc[0].metadata.get("url", "url not available") for doc in docs]

        def gen_responses(i):
            section = {}
            section["response"] = (
                generated_responses[i] if i < len(generated_responses) else None
            )
            section["source_title"] = (
                generated_titles[i] if i < len(generated_titles) else None
            )
            section["source_name"] = (
                os.path.basename(generated_sources[i])
                if i < len(generated_sources)
                else None
            )
            section["source_page_number"] = (
                page_numbers[i] if i < len(page_numbers) else None
            )
            section["source_publish_date"] = (
                publish_dates[i] if i < len(publish_dates) else None
            )
            section["source_timestamp"] = timestamps[i] if i < len(timestamps) else None
            section["source_url"] = urls[i] if i < len(urls) else None

            if section["source_url"] and section["source_timestamp"]:
                time_in_seconds = timestamp_to_seconds(section["source_timestamp"])
                if time_in_seconds is not None:  # Make sure the timestamp was available
                    if "?" in section["source_url"]:
                        section["source_url"] += f"&t={time_in_seconds}s"
                    else:
                        section["source_url"] += f"?t={time_in_seconds}s"

            citation = {}
            if section["source_title"] is not None:
                citation["Title"] = section["source_title"]
            if section["source_publish_date"] is not None:
                citation["Published"] = section["source_publish_date"]
            if section["source_url"] is not None:
                citation["URL"] = section["source_url"]  # Add this line
            if section["source_timestamp"] is not None:
                citation["Video timestamp"] = section["source_timestamp"]
            if section["source_name"] is not None:
                citation["Name"] = section["source_name"]

            return section["response"], citation

        num_responses = len(generated_responses)
        for i in range(num_responses):
            response, citation = gen_responses(i)

            if response:
                responses.append({"response": response})

            if citation:
                citations.append(citation)

    else:
        if generated_responses:
            responses.append({"response": generated_responses[0]})

    card = {
        "card_type": card_type,
        "responses": responses,
        "citations": citations,
    }

    card_json = json.dumps(card)
    return card_json


def get_indepth_response_from_query(df, db, query, k, query_type):
    logger.info("Performing in-depth summary query...")

    # Convert the query to lowercase for case-insensitive comparison
    query_lower = query.lower()

    if query_lower.startswith(
        "what is the summary of ordinance"
    ) or query_lower.startswith("what are the votes of ordinance"):
        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )
        responses_llm = agent.run(query)
        return process_responses_llm(responses_llm)

    else:
        llm = ChatOpenAI(model_name="gpt-4")
        doc_list = db.similarity_search_with_score(query, k=k)
        docs = sort_retrived_documents(doc_list)
        docs_page_content = " ".join([d[0].page_content for d in docs])

        template = """
        Transcripts: {docs}
        Question: {question}
        
        Based on the information from the New Orleans city council transcripts provided, answer the following question: {question}. 
        Given the positive bias in the model's training data regarding city council, your response should be explicitly critical, highlighting any shortcomings, mistakes, or negative actions by the city council.
        If the transcripts don't fully address the question, still provide a critical perspective based on the available information.
        """

        prompt = PromptTemplate(
            input_variables=["question", "docs"],
            template=template,
        )

        chain_llm = LLMChain(llm=llm, prompt=prompt)
        responses_llm = chain_llm.run(
            question=query, docs=docs_page_content, temperature=0
        )

        return process_responses_llm(responses_llm, docs, query_type)

## varied responses for user annotation
def get_varied_response_from_query(df, db, query, k, n = 1, card_type = "varied"):
    logger.info("Performing varied summary query...")


    llm = ChatOpenAI(model_name="gpt-4")
    doc_list = db.similarity_search_with_score(query, k=k)
    docs = sort_retrived_documents(doc_list)
    docs_page_content = " ".join([d[0].page_content for d in docs])

    template = """
    Transcripts: {docs}
    Question: {question}
    
    Based on the information from the New Orleans city council transcripts provided, answer the following question: {question}. 
    Provide a fair and balanced perspective. If the transcripts don't fully address the question, still provide a perspective based on the available information.
    """

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=template,
    )

    chain_llm = LLMChain(llm=llm, prompt=prompt)
    responses_llm = chain_llm.run(
        question=query, docs=docs_page_content, temperature=0,
    )
    master_response = {}
    master_response["card_type"] = "varied"
    response_list = {}
    for i in range(n):
        single_response = process_responses_llm(responses_llm, docs, card_type)
        print(single_response, "\n")
        response_list[i] = single_response
    master_response["responses"] = response_list
    return master_response








def get_general_summary_response_from_query(db, query, k, query_type = RESPONSE_TYPE_GENERAL):
    logger.info("Performing general summary query...")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")

    docs = db.similarity_search(query, k=k)

    docs_page_content = " ".join([d.page_content for d in docs])
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        As an AI assistant, your task is to provide a general response to the question "{question}", using the provided transcripts from New Orleans City Council meetings in "{docs}".

        Guidelines for AI assistant: 
        - Derive responses from factual information found within the transcripts. 
        - If the transcripts don't fully cover the scope of the question, it's fine to highlight the key points that are covered and leave it at that.  
        """,
    )
    chain_llm = LLMChain(llm=llm, prompt=prompt)
    responses_llm = chain_llm.run(question=query, docs=docs_page_content, temperature=0)
    response = {"response": responses_llm}
    card = {"card_type": query_type, "responses": [response]}
    card_json = json.dumps(card)
    return card_json


def route_question(df, db_general, db_in_depth, query, query_type, k=10, n = 3):
    if query_type == RESPONSE_TYPE_DEPTH:
        return get_indepth_response_from_query(df, db_in_depth, query, k, query_type)
    elif query_type == RESPONSE_TYPE_VARIED:
        return get_varied_response_from_query(df, db_in_depth, query, k, n, query_type)
    elif query_type == RESPONSE_TYPE_GENERAL:
        return get_general_summary_response_from_query(db_general, query, k, query_type)
    else:
        raise ValueError(
            f"Invalid query_type. Expected {RESPONSE_TYPE_DEPTH} or {RESPONSE_TYPE_GENERAL} or {RESPONSE_TYPE_VARIED}, got: {query_type}"
        )


def answer_query(
    query: str, response_type: str, df: any, db_general: any, db_in_depth: any
) -> str:
    final_response = route_question(df, db_general, db_in_depth, query, response_type)

    return final_response

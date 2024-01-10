import json
import os
import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from datetime import datetime

from helper import sort_retrived_documents
from api import RESPONSE_TYPE_DEPTH, RESPONSE_TYPE_GENERAL

logger = logging.getLogger(__name__)


def convert_date_format(date_str):
    """Convert date from 'M-D-YYYY' or 'MM-DD-YYYY' to 'MM/DD/YYYY' format."""
    if not isinstance(date_str, str):
        return "Invalid input: not a string"

    if '/' in date_str:
        return date_str

    input_format = "%m-%d-%Y"  

    try:
        date_obj = datetime.strptime(date_str, input_format)
    except ValueError:
        try:
            input_format = "%-m-%-d-%Y"  
            date_obj = datetime.strptime(date_str, input_format)
        except ValueError:
            return "Invalid date format"
        
    return date_obj.strftime("%m/%d/%Y")


def timestamp_to_seconds(timestamp):
    if "timestamp not available" in timestamp:
        return None  # or another default value like -1 or 0
    
    start_time = timestamp.split("-")[0]  # Split by '-' and take the first part
    print(start_time)
    
    time_parts = [int(i) for i in start_time.split(":")]
    
    if len(time_parts) == 3:
        h, m, s = time_parts
    elif len(time_parts) == 2:
        h, m = time_parts
        s = 0 
    else:
        raise ValueError("Invalid timestamp format: " + timestamp)

    return h * 3600 + m * 60 + s


def process_responses_llm(responses_llm, docs=None):
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
            convert_date_format(doc[0].metadata.get("publish_date", "date not available")) for doc in docs     
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
            if section["source_page_number"] is not None:
                citation["Page Number"] = section["source_page_number"]

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
        "card_type": RESPONSE_TYPE_DEPTH,
        "responses": responses,
        "citations": citations,
    }
    card_json = json.dumps(card)
    return card_json


def append_metadata_to_content(doc_list):
    updated_docs = []

    for doc_tuple in doc_list:
        doc, score = doc_tuple  
        metadata = doc.metadata
        publish_date = metadata.get("publish_date")

        if publish_date is not None:
            updated_content = f"Document: {doc.page_content} (Published on: {publish_date})"
        else:
            updated_content = doc.page_content

        updated_doc_info = {
            'content': updated_content,
            'metadata': metadata,
            'score': score
        }
        
        updated_docs.append(updated_doc_info)

    return updated_docs


def transform_query_for_date(query):
    return (
        query
        + "(SYSTEM NOTE: this query related to a specific time period, therefore, you should sort the documents by the publish dates to best answer the query)"
    )


def get_indepth_response_from_query(df, db, query, k):
    logger.info("Performing in-depth summary query...")

    llm = ChatOpenAI(model_name="gpt-4-1106-preview")

    template_date_detection = """
        Analyze the following query: "{query}".
        Does this query pertain to a specific date or time period, or require sorting the city council documents by date? 
        Respond with 'yes' or 'no'.
    """

    prompt_date = PromptTemplate(
        input_variables=["query"],
        template=template_date_detection,
    )
    is_date_related_chain = LLMChain(llm=llm, prompt=prompt_date)

    is_date_related = is_date_related_chain.run(query=query)

    # Modify the query if it is date-related
    if is_date_related.strip().lower() == "yes":
        print("Date related")
        query = transform_query_for_date(query)

    doc_list = db.similarity_search_with_score(query, k=k)

    docs = sort_retrived_documents(doc_list)

    docs_page_content = append_metadata_to_content(docs)

    template = """
    Question: {question}

    ### Bias Guidelines:
    
    Please be aware of inherent biases within the document corpus, especially an overrepresentation of certain types of documents.
    These biases may result in the retrieval of documents that are irrelevant to the question. 
    When analyzing documents to answer the question, it is crucial to critically evaluate their relevance to the question at hand.
    To ensure accuracy and relevance in your analysis you must identify and disregard irrelevant documents by actively identifying documents that, despite being returned by the database, do not substantively address the question.
    Such documents should be disregarded in the analysis.

    ### Response Guidelines:

    Based on the information from the New Orleans city council documents provided, answer the following question: {question}. 
    Your answer must not exceed 5,000 tokens.
    Please provide direct and concise responses without unnecessary verbosity.

    If possible, extract the key points, decisions, and actions discussed during the city council meetings relevant to {question};
    highlight any immediate shortcomings, mistakes, or negative actions by the city council relevant to {question}; 
    elaborate on the implications and broader societal or community impacts of the identified issues relevant to {question};
    investigate any underlying biases or assumptions present in the city council's discourse or actions relevant to {question}. 

    If your response includes technical or uncommon terms related to city council that may not be widely understood, provide a brief definition for those terms at the end of your response in the following format where each definition is on a new line:

    Definitions:
    
    Word: Definition
    
    Word: Definition
    
    Word: Definition

    The final output should be in paragraph form without any formatting, such as prefixing your points with "a.", "b.", or "c."
    The final output should not include any reference to the model's active sorting by date.
    The final output should not include any reference to the publish date. For example, all references to "(published on mm/dd/yyyy)" should be omitted. 

    Documents: {docs}
    """


    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=template,
    )

    chain_llm = LLMChain(llm=llm, prompt=prompt)
    responses_llm = chain_llm.run(question=query, docs=docs_page_content, temperature=1)
    print(responses_llm)

    return process_responses_llm(responses_llm, docs)


def get_general_summary_response_from_query(db, query, k):
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
    card = {"card_type": RESPONSE_TYPE_GENERAL, "responses": [response]}
    card_json = json.dumps(card)
    return card_json


def route_question(df, db_general, db_in_depth, query, query_type, k=20):
    if query_type == RESPONSE_TYPE_DEPTH:
        return get_indepth_response_from_query(df, db_in_depth, query, k)
    elif query_type == RESPONSE_TYPE_GENERAL:
        return get_general_summary_response_from_query(db_general, query, k)
    else:
        raise ValueError(
            f"Invalid query_type. Expected {RESPONSE_TYPE_DEPTH} or {RESPONSE_TYPE_GENERAL}, got: {query_type}"
        )


def answer_query(
    query: str, response_type: str, df: any, db_general: any, db_in_depth: any
) -> str:
    final_response = route_question(df, db_general, db_in_depth, query, response_type)

    return final_response


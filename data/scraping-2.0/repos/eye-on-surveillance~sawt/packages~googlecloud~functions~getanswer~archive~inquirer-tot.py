import logging
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import json
import os

from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

from helper import sort_retrived_documents
from api import RESPONSE_TYPE_DEPTH, RESPONSE_TYPE_GENERAL

logger = logging.getLogger(__name__)

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re


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
        "card_type": RESPONSE_TYPE_DEPTH,
        "responses": responses,
        "citations": citations,
    }
    card_json = json.dumps(card)
    return card_json


def timestamp_to_seconds(timestamp):
    if "timestamp not available" in timestamp:
        return None
    start_time = timestamp.split("-")[0]
    time_parts = start_time.split(":")
    h, m, s = 0, 0, 0
    if len(time_parts) == 3:
        h, m, s = [int(i) for i in time_parts]
    elif len(time_parts) == 2:
        h, m = [int(i) for i in time_parts]
    elif len(time_parts) == 1:
        m = int(time_parts[0])
    return h * 3600 + m * 60 + s


def create_agent(df):
    return create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k"),
        df,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )


def evaluate_document_relevance(llm, docs, query, threshold):
    template = """
    Transcripts: {docs}
    Given the documents retrieved for the query: {query}, rate the relevance and quality of these documents for answering the query on a scale of 1 to 10. 
    Please provide one score for all documents in the following format: confidence_score: score
    A:
    """
    prompt = PromptTemplate(input_variables=["docs", "query"], template=template)
    chain = LLMChain(llm=llm, prompt=prompt, output_key="confidence_score")
    result = chain.run(docs=docs, query=query, temperature=0)

    result_dict = {"confidence_score": float(result.split(":")[1].strip())}
    confidence_score = result_dict.get("confidence_score", 0)

    # print(f"Query: {query}, Result: {result}")

    try:
        confidence_score = int(confidence_score)
    except ValueError:
        logging.warning(
            f"Could not convert confidence score to an integer: {confidence_score}"
        )
        confidence_score = 0

    better_query_needed = confidence_score < threshold

    return {
        "confidence_score": confidence_score,
        "better_query_needed": better_query_needed,
    }


def generate_better_query(llm, original_query, docs, threshold):
    # Template for generating a better query
    template = """
    Transcripts: {docs}
    The original query: {original_query} did not yield satisfactory results with a confidence below {threshold}.
    Please provide a new and improved query that we can send to the faiss vector database for document retrival.
    The new query must aim to answer the {original_query}
    A:
    """

    prompt = PromptTemplate(
        input_variables=["original_query", "docs", "threshold"], template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="better_query")
    result = chain.run(
        original_query=original_query, docs=docs, threshold=threshold, temperature=0
    )
    if result:
        better_query = result
    else:
        logging.warning("Result is empty. Using original query instead.")
        better_query = original_query
    return better_query


def refine_query(db, llm, query, k, threshold_db, sort_retrieved_documents):
    iteration_counter = 0
    max_iterations = 1
    query_scores = {}
    updated_query = query

    # Evaluate the initial query and store its score
    doc_list = db.similarity_search_with_score(updated_query, k=k)
    docs = sort_retrieved_documents(doc_list)
    evaluation_result = evaluate_document_relevance(
        llm, docs, updated_query, threshold_db
    )
    confidence_rating = evaluation_result.get("confidence_score", 0)

    # Store the query if its score is not already in the dictionary
    if confidence_rating not in query_scores:
        query_scores[confidence_rating] = updated_query

    while iteration_counter < max_iterations and confidence_rating < threshold_db:
        # If the initial query did not meet the threshold, refine it
        updated_query = generate_better_query(llm, updated_query, docs, threshold_db)

        doc_list = db.similarity_search_with_score(updated_query, k=k)
        docs = sort_retrieved_documents(doc_list)
        evaluation_result = evaluate_document_relevance(
            llm, docs, updated_query, threshold_db
        )
        confidence_rating = evaluation_result.get("confidence_score", 0)

        # Store the query if its score is not already in the dictionary
        if confidence_rating not in query_scores:
            query_scores[confidence_rating] = updated_query

        iteration_counter += 1

    highest_score = max(query_scores.keys())
    best_query = query_scores[highest_score]
    return best_query


def run_vector_search(db, best_query_vector_db, k, sort_retrieved_documents):
    doc_list = db.similarity_search_with_score(best_query_vector_db, k=k)
    docs = sort_retrieved_documents(doc_list)
    docs_page_content = " ".join([d[0].page_content for d in docs])
    return docs, docs_page_content, best_query_vector_db


def ensure_dict(obj):
    if isinstance(obj, dict):
        return obj
    elif isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            logging.warning(f"Could not convert string to dictionary: {obj}")
            return {}
    else:
        logging.warning(f"Object is not a dictionary or string: {obj}")
        return {}


def parse_angles(output_str):
    # Split string based on pattern (e.g., "1. ", "2. ", etc.)
    angle_sections = re.split(r"\d+\.\s", output_str)[1:]

    angles = {}
    for index, section in enumerate(angle_sections, start=1):
        angles[str(index)] = section.strip()
    return angles


def generate_synthesized_angle(llm, angles_dict, confidence_dict, docs):
    # Combine angles and their confidence ratings
    combined_angles = "\n".join(
        [
            f"Angle {i+1} (Confidence: {confidence_dict.get(angle, 'Unknown')}): {angle}"
            for i, angle in enumerate(angles_dict.values())
        ]
    )

    # Template for generating a synthesized angle
    template = """
    Transcripts: {docs}
    Review the following brainstormed angles along with their confidence ratings:
    {combined_angles}
    
    Identify the most insightful and relevant aspects from each brainstormed angle, while also considering their confidence ratings. Reinterpret, combine, or expand on these ideas to form a cohesive and improved approach for analyzing the transcripts. Please synthesize these angles into a new, comprehensive angle in the following format:
    
    Angle: ... 
    
    A:
    """

    prompt = PromptTemplate(
        input_variables=["combined_angles", "docs"], template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt, output_key="synthesized_angle")
    result = chain.run(combined_angles=combined_angles, docs=docs, temperature=0)
    if result:
        synthesized_angle = result
    else:
        logging.warning("Result is empty. Using default synthesized angle instead.")
        synthesized_angle = (
            "A new angle could not be synthesized based on the provided input."
        )

    return synthesized_angle


def get_indepth_response_from_query(
    df,
    db,
    query,
    k,
    max_iterations=1,
):
    logger.info("Performing in-depth summary query...")

    query_lower = query.lower()
    if query_lower.startswith(
        ("list the votes for ordinance", "what were the votes for ordinance")
    ):
        agent = create_agent(df)
        responses_llm = agent.run(query)
        return process_responses_llm(responses_llm)

    else:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

        iteration_counter_db = 0
        confidence_rating_db = 0
        threshold_db = 10

        ## Stage 1: Query refinement stage. Task: evaluate the relevance of docs returned from vector db with respect to query
        best_query = refine_query(
            db, llm, query, k, threshold_db, sort_retrived_documents
        )
        docs, docs_page_content, best_query_vector = run_vector_search(
            db, best_query, k, sort_retrived_documents
        )

        # print(best_query_vector)

        ## Helper funcs
        def execute_brainstorming_stage(docs):
            template1 = """
            Transcripts: {docs}
            Question: {question}

            To provide a detailed and accurate response based on the transcripts provided, brainstorm three distinct strategies that leverage the specific content and context of these documents. Focus on methodologies that utilize the information within the transcripts to clarify, explore, and elaborate on the query. 

            Please provide the output in the following format:
            1. [Detailed strategy emphasizing analysis, interpretation, or cross-referencing within the transcripts themselves]
            2. [Another strategy that relies on extracting and building upon specific details, examples, or discussions found in the transcripts]
            3. [A third strategy that uses the thematic or contextual elements present in the transcripts to provide an in-depth understanding of the topic]

            A:
            """

            prompt1 = PromptTemplate(
                input_variables=["question", "docs"], template=template1
            )
            chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="angles")
            responses_llm = chain1.run(question=best_query, docs=docs, temperature=1)
            # print(f"Angle: {responses_llm}")

            # print("Raw brainstorming response:", responses_llm)
            parsed_angles = parse_angles(responses_llm)
            # print("Parsed angles:", parsed_angles)
            return parsed_angles

        def evaluate_output(angles, docs, angle_confidence_dict):
            template1_evaluation = """
            Transcripts: {docs}
            Based on the brainstormed angles: {angles}, how confident are you in the quality and relevance of these perspectives for the query: {question}? 
            Rate your confidence on a scale of 1 to 10. Only provide the number.
            A:
            """
            prompt1_evaluation = PromptTemplate(
                input_variables=["question", "docs", "angles"],
                template=template1_evaluation,
            )
            chain1_evaluation = LLMChain(
                llm=llm, prompt=prompt1_evaluation, output_key="confidence_rating"
            )

            for angle, content in angles.items():
                result = chain1_evaluation.run(
                    question=best_query, docs=docs, angles=content, temperature=0
                )
                print(f"Angle: {angle}, Content: {content}, Result: {result}")

                if isinstance(result, (int, float)):
                    confidence_rating = result
                elif isinstance(result, str):
                    try:
                        confidence_rating = int(result)
                    except ValueError:
                        confidence_rating = 0
                elif isinstance(result, dict) and "confidence_rating" in result:
                    confidence_rating = result["confidence_rating"]
                else:
                    confidence_rating = 0

                angle_confidence_dict[content] = confidence_rating
                # print(f"Content: {content}, Confidence: {confidence_rating}")

            # print(f"DEBUG: angles before check = {angles}")
            if not angle_confidence_dict or all(
                v == 0 for v in angle_confidence_dict.values()
            ):
                logging.warning(
                    "No angles were evaluated or all angles have zero confidence. Returning the first angle."
                )
                best_angle = list(angles.keys())[0]  # Get the first angle
                return {"best_angle": best_angle, "confidence_rating": 0}

            # Sorting the dictionary by values. In case of a tie, the first item with the maximum value will be chosen.
            best_angle = max(angle_confidence_dict, key=angle_confidence_dict.get)
            # print(f"Best Angle: {best_angle}")

            return {
                "best_angle": best_angle,
                "confidence_rating": angle_confidence_dict[best_angle],
                "angle_confidence_dict": angle_confidence_dict,
            }

        ### Stage 2: Evaluate angles returned. Choose the best angle.

        threshold_brainstorm = 10
        iteration_counter_brainstorm = 0
        confidence_rating_brainstorm = 0

        ### Iterate over the brainstorming function until an appropriate angle is found:

        # Brainstorming includes: I have a query related to the New Orleans city council about {question}.
        # Could you brainstorm three distinct angles or perspectives to approach this query
        # Based on the brainstormed angles: {angles}, how confident are you in the quality and relevance of these perspectives for the query: {question}?
        # Rate your confidence on a scale of 1 to 10.

        angle_confidence_dict = {}
        while (
            confidence_rating_brainstorm < threshold_brainstorm
            and iteration_counter_brainstorm < max_iterations
        ):
            logging.info("Brainstorming function invoked.")
            angles_dict = execute_brainstorming_stage(docs_page_content)
            response = evaluate_output(angles_dict, docs, angle_confidence_dict)
            confidence_rating_brainstorm = int(response.get("confidence_rating", 0))
            angle_confidence_dict.update(
                response.get("angle_confidence_dict", {})
            )  # Cumulatively updating the dictionary

            iteration_counter_brainstorm += 1
            logging.info(
                f"Iteration: {iteration_counter_brainstorm}, Confidence Rating: {confidence_rating_brainstorm}"
            )

        if iteration_counter_brainstorm == max_iterations:
            logging.warning(
                f"Maximum number of iterations ({max_iterations}) reached without crossing the confidence threshold. Brainstorm func will no longer be re-run."
            )

        best_angle = max(angle_confidence_dict, key=angle_confidence_dict.get)
        print(f"Best Angle: {best_angle}")

        # Stage 2: Initial Analysis Stage
        template2 = """
        Using the selected approach: {angle}, and the documents: {docs} as references:
            a. Extract the key points, decisions, and actions discussed during the city council meetings relevant to {question}.
            b. Highlight any immediate shortcomings, mistakes, or negative actions by the city council relevant to {question}.
        A:
        """

        prompt2 = PromptTemplate(
            input_variables=["question", "docs", "angle"], template=template2
        )
        chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="evaluated_approaches")

        # Stage 3: Deeper Analysis Stage
        template3 = """
        Transcripts: {docs}
        Question: {question}

        Building upon the initial analysis and based on the selected angle from {evaluated_approaches}, engage in a deeper examination:
            a. Elaborate on the implications and broader societal or community impacts of the identified issues.
            b. Investigate any underlying biases or assumptions present in the city council's discourse or actions relevant to {question}.
        A:
        """
        prompt3 = PromptTemplate(
            input_variables=["question", "docs", "evaluated_approaches"],
            template=template3,
        )
        chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="deepen_thought_process")

        # Stage 4: Synthesis
        template4 = """
        Transcripts: {docs}
        With the output from your deeper analysis stage: {deepen_thought_process}, use the transcripts to synthesize your findings in the following manner:
            a. Identify and draw connections between the discussed points, examining any patterns of behavior or recurrent issues relevant to {question}.
            b. Offer a critical perspective on the city council's actions or decisions related to {question}, utilizing external knowledge if necessary. Highlight any inconsistencies or contradictions.
            c. Summarize the critical insights derived from the analysis regarding {question}.
        A:
        """
        prompt4 = PromptTemplate(
            input_variables=["question", "docs", "deepen_thought_process"],
            template=template4,
        )
        chain4 = LLMChain(llm=llm, prompt=prompt4, output_key="ranked_insights")

        # Connecting the chains
        overall_chain = SequentialChain(
            chains=[chain2, chain3, chain4],
            input_variables=["question", "docs", "angle"],
            output_variables=["ranked_insights"],
            verbose=True,
        )

        responses_llm = overall_chain.run(
            question=best_query, docs=docs_page_content, angle=best_angle, temperature=0
        )
        # print(best_angle)
        # print(best_query)
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


def route_question(df, db_general, db_in_depth, query, query_type, k=10):
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

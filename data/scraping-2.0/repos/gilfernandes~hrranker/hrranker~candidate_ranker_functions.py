## Deprecated code. First try using directly OpenAI functions

from pydantic.tools import parse_obj_as
from langchain.schema import Document

from typing import List, Any

import openai
import json

from hrranker.hr_model import (
    CandidateInfo,
    NameOfCandidateResponse,
    NumberOfYearsResponse,
    NumberOfYearsResponseWithWeight,
    parse_name_of_candidate_json,
    parse_number_of_year_response_json,
    sort_candidate_infos,
    name_of_candidate_response_schema,
    number_of_years_response_schema,
    number_of_years_description,
)
from hrranker.extract_data import extract_data
from hrranker.config import cfg
from hrranker.config import cfg
from hrranker.log_init import logger

openai.api_key = cfg.openai_api_key


def process_questions(questions_schemas: List[Any], doc: Document) -> CandidateInfo:
    name_of_candidate_response: NameOfCandidateResponse = None
    number_of_year_responses: List[NumberOfYearsResponseWithWeight] = []
    for question_schema in questions_schemas:
        function_name = "get_answer_for_user_query"
        question = f"""Given the following extracted parts of a long document and a question, create a final answer. 

QUESTION: {question_schema["question"]}
=========
{doc.page_content}
=========
"""
        question_schema_class = question_schema["class"]
        response = openai.ChatCompletion.create(
            model=cfg.model,
            messages=[{"role": "user", "content": question}],
            functions=[
                {
                    "name": function_name,
                    "description": question_schema["description"],
                    "parameters": question_schema["schema"],
                }
            ],
            function_call={"name": function_name},
        )
        response_json = json.loads(
            response.choices[0]["message"]["function_call"]["arguments"]
        )
        logger.info(f"response_json: {response_json}")
        if question_schema_class == NameOfCandidateResponse:
            name_of_candidate_response = parse_name_of_candidate_json(response_json)
        elif question_schema_class == NumberOfYearsResponse:
            number_of_year_response = parse_number_of_year_response_json(
                response_json, question_schema
            )
            year_weight = question_schema["year_weight"]
            number_of_year_responses.append(
                NumberOfYearsResponseWithWeight(
                    number_of_years_response=number_of_year_response,
                    score_weight=year_weight,
                )
            )

    return CandidateInfo(
        name_of_candidate_response=name_of_candidate_response,
        number_of_years_responses=number_of_year_responses,
        source_file=doc.metadata["source"],
    )


questions_schemas: List[Any] = [
    {
        "question": "Which is the name, age and gender of the candidate?",
        "schema": name_of_candidate_response_schema,
        "class": NameOfCandidateResponse,
        "description": "Get user answer or reply with 0 for age or 'unknown' for name and gender if you do not know",
    },
    {
        "question": "How many years of experience with Wordpress does this candidate have?",
        "schema": number_of_years_response_schema,
        "class": NumberOfYearsResponse,
        "description": number_of_years_description,
        "year_weight": 3,
    },
    {
        "question": "How many years of experience with PHP development does this candidate have?",
        "schema": number_of_years_response_schema,
        "class": NumberOfYearsResponse,
        "description": number_of_years_description,
        "year_weight": 2,
    },
    {
        "question": "How many years of experience with Javascript development does this candidate have?",
        "schema": number_of_years_response_schema,
        "class": NumberOfYearsResponse,
        "description": number_of_years_description,
        "year_weight": 1,
    },
]


if __name__ == "__main__":
    path = cfg.doc_location
    docs = documents = extract_data(path)
    candidate_infos: List[CandidateInfo] = []
    for doc in docs:
        logger.info(f"source: {doc.metadata}")
        candidate_info: CandidateInfo = process_questions(questions_schemas, doc)
        logger.info(f"candidate_info: {candidate_info}")
        candidate_infos.append(candidate_info)
    candidate_infos = sort_candidate_infos(candidate_infos)

    logger.info(f"Candidate ranking:")
    logger.info(f"------------------")
    for candidate in candidate_infos:
        if candidate.name_of_candidate_response:
            logger.info(
                f"Candidate: {candidate.name_of_candidate_response.name}: {candidate.score} points ({candidate.source_file})"
            )

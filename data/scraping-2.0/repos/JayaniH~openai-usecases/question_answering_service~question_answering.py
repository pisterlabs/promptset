import logging

import numpy as np
import openai
from flask import current_app as app
from flask import request

from question_answering_service import document_embeddings, documents, tokenizer
from question_answering_service.constants import COMPLETIONS_API_PARAMS, MAX_SECTION_LEN, SEPARATOR, SEPARATOR_LEN, \
    HTTP_500_INTERNAL_SERVER_ERROR
from question_answering_service.utils import get_embedding, count_tokens


@app.route('/generate_answer')
def generate_answer():
    try:
        question = request.args.get('question')
        prompt = construct_prompt(question)
        response = openai.Completion.create(
            prompt=prompt,
            **COMPLETIONS_API_PARAMS
        )
        answer = response['choices'][0]['text'].strip(" \n")
        return answer

    except Exception as e:
        logging.error("Error generating answer: " + str(e), exc_info=True)
        return "ERROR: There was an error generating the answer.", HTTP_500_INTERNAL_SERVER_ERROR


def get_document_similarity(query):
    try:
        query_embedding = get_embedding(query)

        document_similarities = sorted([
            (np.dot(np.array(query_embedding), np.array(doc_embedding)), doc_index) for doc_index, doc_embedding in
            document_embeddings.items()
        ], reverse=True)

        return document_similarities

    except Exception as e:
        logging.error("Error computing document similarity: " + str(e), exc_info=True)


def construct_prompt(question):
    most_relevant_document_sections = get_document_similarity(question)

    if not most_relevant_document_sections:
        raise Exception("Error computing document similarity for question: " + question)

    chosen_sections = []
    chosen_sections_len = 0

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = documents[section_index]
        chosen_sections_len += count_tokens(document_section, tokenizer) + SEPARATOR_LEN
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + document_section.replace("\n", " "))

    instruction = "Answer the question as truthfully as possible using the provided context, and if the answer is " \
                  "not contained within the text below, say \"I don't know.\"\n\n"

    return instruction + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

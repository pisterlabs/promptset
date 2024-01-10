"""This file contains the correctors services responsible for the correction of the exams by the AI."""

import json

import connectors.openai_connector as openai_connector
import connectors.mongodb_connector as mongodb_connector
from schemas.items import QuestionDoc, ImageDoc, CorrectionSchema
from schemas.prompts_schemas import BinaryCorrectionPrompt
from utils.parsers import json_parser

"""
This file is under construction. In order to get a function that corrects the questions,
I need to get access for firebase url storage, and then then function should: 
1. receive as argument a question document, and an image document.
2. form the prompt to send to openai.
3. parse the response from openai.
4. save the response on the database.
"""

def correct_question(question: QuestionDoc, image: ImageDoc):
    """
    Constructs the prompts to correct a question, sends it to openai, and parses the response.

    Returns:
     ImageDoc: The corrected image document

    """

    prompt = BinaryCorrectionPrompt(
        question_text=question.question_text,
        question_answer=question.question_answer
    )

    params = {
        "system_message": prompt.system_prompt,
    }

    response = openai_connector.send_online_image_to_openai(prompt.template, image.url, **params)

    parsed_response = json_parser(response["content"])

    correction_schema = CorrectionSchema(
        is_correct=parsed_response["is_correct"],
        correction_tokens={
            "completion_tokens": response["completion_tokens"],
            "prompt_tokens": response["prompt_tokens"],
            "total_tokens": response["total_tokens"]
        }
        )
    
    if "correction_comments" in parsed_response:
        correction_schema.correction_comments = parsed_response["correction_comments"]

    image.correction_results = correction_schema

    image_updater = mongodb_connector.update_image_doc(image)

    return image

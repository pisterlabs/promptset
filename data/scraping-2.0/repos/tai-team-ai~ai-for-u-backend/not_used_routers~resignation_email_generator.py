"""
Module for the resignation email generator lambda function.

Module contains models and post endpoint for the resignation email generator lambda function. 
The lambda function uses the openai model text-davinci-003 to generate a resignation email. 
A prompt is generated from the request and sent to openai for processing. The response from 
openai is then returned to the client.

Attributes:
    router (APIRouter): Router for the lambda function.
    ResignationEmailGeneratorModel (AIToolModel): Model for the request.
    ResignationEmailGeneratorResponseModel (AIToolModel): Model for the response.
    get_openai_response (function): Method to get response from openai.
    resignation_email_generator (function): Post endpoint for the lambda function.
    
"""

from datetime import datetime
import logging
import os
import sys
from typing import Optional, List
from fastapi import APIRouter, Response, status,Request
from pydantic import conint, constr
import openai
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../utils"))
from utils import initialize_openai, prepare_response, AIToolModel, sanitize_string

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

router = APIRouter()

class ResignationEmailGeneratorModel(AIToolModel):
    reason: constr (min_length=1, max_length=400)
    resignation_date: datetime
    company_name: Optional[constr (min_length=0, max_length=70)] = ""
    manager_name: Optional[constr (min_length=0, max_length=20)] = ""
    notes: Optional[constr (min_length=0, max_length=250)] = ""
    bluntness: Optional[conint(ge=0, le=100)] = 50

class ResignationEmailGeneratorResponseModel(AIToolModel):
    resignation_email: str = ""

def get_openai_response(prompt: str, bluntness: int=50) -> str:
    """
    Method uses openai model text-davinci-003 to generate resignation email.

    The bluntness parameter is used to control the temperature of the model and the 
    frequency penalty. For temperature, the bluntness is divided by 100 and the value is mapped
    to the range [0.1, 0.4]. For frequency penalty, the bluntness is divided by 100 and the value
    is mapped to the range [0.2, 0.5]. The prompt is sent to openai for processing. The response 
    from openai is then returned to the client.

    :param prompt: Request containing notes and options for summarization.

    :return: response to prompt
    """
    initialize_openai()
    prompt_len = len(prompt)
    max_tokens = min(500, prompt_len)
    temperature = 0.1 + (1 - bluntness / 100) * 0.9
    frequency_penalty = 0.2 + (bluntness / 100) * 0.3
    logger.info(f"prompt: {prompt}")
    logger.info(f"temperature: {temperature}")
    logger.info(f"frequency_penalty: {frequency_penalty}")
    logger.info(f"max_tokens: {max_tokens}")
    openai_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=min(temperature, 1.0),
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=min(frequency_penalty, 2.0),
        presence_penalty=0,
        stream=False,
        logprobs=None,
        echo=False,
    )
    logger.info(f"openai_response: {openai_response}")
    return openai_response.choices[0].text.strip()

@router.post("/resignation_email_generator", response_model=ResignationEmailGeneratorResponseModel, status_code=status.HTTP_200_OK)
async def resignation_email_generator(resignation_email_generator_model: ResignationEmailGeneratorModel, response: Response, request: Request):
    """
    Post endpoint for the lambda function.

    The post endpoint uses the openai model text-davinci-003 to generate a resignation email. 
    A prompt is generated from the request and sent to openai for processing. The response from 
    openai is then returned to the client.

    :param resignation_email_generator_model: Request containing notes and options for summarization.
    :param response: Response to the request.

    :return: response to prompt
    """
    logger.info(f"resignation_email_generator_model: {resignation_email_generator_model}")
    prepare_response(response, request)
    for field_name, field_value in resignation_email_generator_model:
        if isinstance(field_value, str):
            field_value = sanitize_string(field_value)
        if field_value == "":
            setattr(resignation_email_generator_model, field_name, "None")
    prompt = "You are an email writing robot. Specifically you are designed to write resignation emails. "\
        "I will provide a resignation reason, resignation date, and my company name and you will generate a "\
            "resignation email. I may optionally provide my managers name and some additional notes that I would "\
                "like included in the resignation email. I will direct you on how blunt I would like the email to be, "\
                    "where 1 is the least blunt and 100 is the most blunt. You should include in the email a questions that "\
                        "asks about how to make the transition go smoothly for the team. The following is the information for "\
                            "the first resignation email I would like you to write:\n\n"
    prompt += f"Reason for resignation: \n{resignation_email_generator_model.reason}"
    prompt += f"\n\nResignation date: \n{resignation_email_generator_model.resignation_date.strftime('%m/%d')}"
    prompt += f"\n\nCompany name: \n{resignation_email_generator_model.company_name}"
    prompt += f"\n\nManager name: \n{resignation_email_generator_model.manager_name}"
    prompt += f"\n\nNotes: \n{resignation_email_generator_model.notes}"
    prompt += f"\n\nBluntness: \n{resignation_email_generator_model.bluntness}"

    response_dict = {
        "resignation_email": get_openai_response(prompt, resignation_email_generator_model.bluntness)
    }

    resignation_email_generator_response_model = ResignationEmailGeneratorResponseModel(**response_dict)

    logger.info(f"resignation_email_generator_response_model: {resignation_email_generator_response_model}")
    return resignation_email_generator_response_model



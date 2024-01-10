"""
Module defines the router for the sales inquiry email generator. This endpoint uses the openai 
text-davinci-003 model to generate a sales inquiry email. The endpoint accepts a request containing:
companyName, pointOfContact, name, reason, productName, productDescription, and the problem the product is 
solving. The endpoint then generates a prompt from the request and sends it to openai for processing. 
The response from openai is then returned to the client.

Attributes:
    router (APIRouter): Router for the lambda function.
    SalesInquiryEmailGeneratorModel (AIToolModel): Model for the request.
    SalesInquiryEmailGeneratorResponseModel (AIToolModel): Model for the response.
    get_openai_response (function): Method to get response from openai.
    sales_inquiry_email_generator (function): Post endpoint for the lambda function.
"""

from logging import Logger
import openai
import os
import sys
from typing import Optional
from fastapi import APIRouter, Response, status, Request
from pydantic import conint, constr
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../utils"))
from utils import initialize_openai, prepare_response, AIToolModel, sanitize_string

router = APIRouter()

class SalesInquiryEmailGeneratorModel(AIToolModel):
    companyName: constr (min_length=1, max_length=70)
    pointOfContact: constr (min_length=1, max_length=70) = "None"
    name: constr (min_length=1, max_length=70) = "None"
    reason: constr (min_length=1, max_length=400) = "None"
    productName: constr (min_length=1, max_length=70) = "None"
    productDescription: constr (min_length=1, max_length=400) = "None"
    problem: constr (min_length=1, max_length=400) = "None"

class SalesInquiryEmailGeneratorResponseModel(AIToolModel):
    sales_inquiry_email: constr (min_length=1, max_length=10000)

def get_openai_response(prompt: str, num_emails_to_generate: int=1) -> str:
    """
    Method to get response from openai.

    Args:
        prompt (str): Prompt to send to openai.

    Returns:
        str: Response from openai.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=600,
        temperature=0.8,
        top_p=1,
        frequency_penalty=1.19,
        presence_penalty=0.96,
        n=num_emails_to_generate
    )

    return response["choices"][0]["text"]

@router.post("/sales-inquiry-email-generator", response_model=SalesInquiryEmailGeneratorResponseModel)
async def sales_inquiry_email_generator(request_model: SalesInquiryEmailGeneratorModel, request: Request, response: Response):
    """
    Post endpoint for the lambda function.

    Args:
        request (SalesInquiryEmailGeneratorModel): Request from the client.
        response (Response): Response to the client.

    Returns:
        SalesInquiryEmailGeneratorResponseModel: Response to the client.
    """
    prepare_response(response, request)
    initialize_openai()
    prompt = f"""
    You are a professional sales inquiry email generator. 
    You are really good at generating friendly and professional emails that present 
    a product that solves the recipients problem. Your email should invoke further 
    conversation from the recipient of the email so I can build a relationship with them. 
    I will provide you the company name the email is sent to, the point of contact at the company, 
    my name, the reason for my inquiry, the product I am selling, the description of the product 
    I am selling, and the problem that my product solves. If I don't provide the previous information,
    I will designation the field with 'None', meaning this information should not be included. 
    The information for the first email you should generate is below.

    Company Name:\n{request_model.companyName}
    Point of Contact:\n{request_model.pointOfContact}
    My Name: \n{request_model.name}
    Reason: \n{request_model.reason}
    Product Name: \n{request_model.productName}
    Product Description: \n{request_model.productDescription}
    Problem: \n{request_model.problem}

    Sales Inquiry Email:
    """
    prompt = sanitize_string(prompt)
    open_ai_response = get_openai_response(prompt).strip()
    response_model = SalesInquiryEmailGeneratorResponseModel(sales_inquiry_email=open_ai_response)
    return response_model
"""
module for API prediction users
"""

import os
from datetime import date

import openai

from fastapi import APIRouter
from pydantic import BaseModel

openai.api_key = os.environ.get("OPANAI_KEY")

router = APIRouter(
    tags=["OpenAI Service"],
    responses={404: {"description": "Not found in"}},
)



class PredictInformationBody(BaseModel):
    """
    this class is prediction information for predict service
    """
    full_name: str
    isTh: bool
    birthday: date



@router.post("/predict", status_code=200, response_model=str)
def preict_destiny(prediction_information: PredictInformationBody):
    """
    this function try to predict destiny from uers
    """

    prompt = f"""Predict the destiny of 
                {prediction_information.full_name}, 
                 who was born on {prediction_information.birthday}. 
                 {'in Thai language' if prediction_information.isTh else ' '}.
            """
    instruction = "You are a fortune teller who predicts fortunes based on historical data. The information you answer will be a powerful message to live the listener."
 
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content" : instruction},
            {"role": "user", "content": prompt},
        ],
    )

    assistant_reply = result['choices'][0]['message']['content']
    return assistant_reply

from typing import List
from app.model import Message
import openai
import os
import json
from app.utils import logger, remove_spanish_special_characters
from datetime import datetime, timedelta, date


openai.api_key = os.environ.get('OPENAI_API_KEY')


json_fn = {
    "name": "calculate_business_info",
    "description": "Gathers all the available properties based on the location, bnbot_id, business name or business owner.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The name of the city or location where the property is."
            },
            "bnbot_id": {
                "type": "string",
                "description": "The ID of the business the user is looking for. The id does not have spaces and always starts with the @ character. i.e: '@casa.de.alejandro'."
            },
            "business_name": {
                "type": "string",
                "description": "The name of the business the user is looking for. This could be a sentence like: 'Hermoso apartamento en edificio centrico en Mercedes."
            },
            "business_owner": {
                "type": "string",
                "description": "The name of the business the owner. This is a person's name like: Gonzalo Sarasua or just Gonzalo."
            }
        },
        "required": []
    }
}

class BusinessSearchDataExtractor:

    def _have_enough_info(self, fn_parameters: dict):
        if "bnbot_id" in fn_parameters and fn_parameters["bnbot_id"] != "":
            return True
        
        if "location" in fn_parameters and fn_parameters["location"] != "":
            fn_parameters["location"] = remove_spanish_special_characters(fn_parameters["location"])
            if "business_name" in fn_parameters and fn_parameters["business_name"] != "":
                return True
            if "business_owner" in fn_parameters and fn_parameters["business_owner"] != "":
                return True
        return False

    def run(self, messages: List[Message]):
        
        messages_input = [{"role": "system", "content": "What are the available properties that the user is looking for?"}]
        for msg in messages:
            messages_input.append({"role": msg.role, "content": msg.text})
        # messages_input.append("role")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages_input,
            functions=[json_fn],
            # function_call={"name": "calculate_business_info"},
            temperature=0., 
            max_tokens=500, 
        )
        if "function_call" in response.choices[0].message and "arguments" in response.choices[0].message["function_call"]:
            fn_parameters = json.loads(response.choices[0].message["function_call"]["arguments"])
            fn_parameters["have_enough_info"] = self._have_enough_info(fn_parameters)
            logger.debug(f"calculate_business_info fn_parameters {fn_parameters}")
            return fn_parameters
        
        return {"have_enough_info": False}
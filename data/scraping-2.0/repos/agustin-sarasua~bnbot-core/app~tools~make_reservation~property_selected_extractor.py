from typing import List
from app.model import Message
import openai
import os
import json
from app.utils import logger
from datetime import datetime, timedelta

openai.api_key = os.environ.get('OPENAI_API_KEY')

json_fn = {
    "name": "set_property_selected",
    "description": "Saves the property information that the user has chosen.",
    "parameters": {
        "type": "object",
        "properties": {
            "property_id": {
                "type": "string",
                "description": "The property_id of the property that the user has chosen."
            },
        },
        "required": []
    }
}

system_message = """Save the property that the user has chosen. 
Here is the information fo the properties:
{properties_info}"""

class PropertySelectedExtractor:

    def run(self, messages: List[Message], properties_info: str):
        
        formatted_system_message = system_message.format(properties_info=properties_info)
        
        messages_input = [{"role": "system", "content": formatted_system_message}]
        for msg in messages:
            messages_input.append({"role": msg.role, "content": msg.text})
        # messages_input.append("role")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages_input,
            functions=[json_fn],
            temperature=0., 
            max_tokens=500, 
        )
        if "function_call" in response.choices[0].message and "arguments" in response.choices[0].message["function_call"]:
            fn_parameters = json.loads(response.choices[0].message["function_call"]["arguments"])
            fn_parameters["user_has_selected"] = ("property_id" in fn_parameters and fn_parameters["property_id"] != "")
            logger.debug(f"set_property_selected fn_parameters {fn_parameters}")
            return fn_parameters
        
        return {"user_has_selected": False }
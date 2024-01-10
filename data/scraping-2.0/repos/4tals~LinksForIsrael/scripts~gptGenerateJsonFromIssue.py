import logging
import os
import re

import openai
import yaml

logging.basicConfig(level=logging.INFO)

class GPTAssistant:
    def __init__(self, api_key: str = None, model: str = 'gpt-4'):
        self.model = model

        openai.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if openai.api_key is None:
            raise ValueError("No API key provided. Set the OPENAI_API_KEY environment variable or pass the api_key argument.")

    def ask_gpt_for_initiative_json(self, issue_body: str) -> str:
        query =  f"""
        Please provide arguments for the add_new_initiative function using the following guidelines:

        * Ensure that no parameter has an empty, null or "no response" value.
        * Add hebrew tags about the relevant initiative, the tags should aid in efficient searching.
        * If possible, provide the "displayName" and "description" fields in Hebrew.

        The details of the new initiative to be added have been provided in the following form, provided in markdown syntax:
        {issue_body}
        """
        logging.info("GPT query: " + query)

        messages = [
            {"role": "system", "content": "You are only used for function calling, and must exclusively use the function you have been provided with"},
            {"role": "user", "content": query}
        ]

        with open(os.environ["GITHUB_WORKSPACE"] + "/.github/ISSUE_TEMPLATE/new_initiative.yml", "r") as stream:
            new_initiative_form = yaml.safe_load(stream)

        # initialize with properties that require special handling
        new_initiative_params = {
            # we don't ask for the display name in the form (and therefore it would not be present in the issue template yaml parsed above)
            "displayName": { 
                "type": "string", 
                "description": "The text displayed for the initiative (defaults to the initiative's name)" 
            }
        }

        arrayParams = {"tags"}
        for field in new_initiative_form["body"]:
            
            field_id = field["id"]
            field_attributes = field["attributes"]
            field_description = field_attributes["label"]

            if field_id in arrayParams:
                new_initiative_params[field_id] = { 
                    "type": "array", 
                    "items": { "type": "string" },
                    "description": field_description
                }
                continue

            if field["type"] == "dropdown": 
                new_initiative_params[field_id] = { 
                    "type": "string", 
                    "enum": field_attributes["options"],
                    "description": field_description
                }
                continue

            new_initiative_params[field_id] = { 
                    "type": "string", 
                    "description": field_description
            }

        functions = [
        {
            "name": "add_new_initiative",
            "description": "Add a new initiative to the database",
            "parameters": {
                "type": "object",
                "properties": new_initiative_params
            },
        }]

        logging.info("Calling OpenAPI with functions: {0}".format(functions))

        response = openai.ChatCompletion.create(
            model = self.model,
            messages = messages,
            functions = functions,
            function_call = {"name": "add_new_initiative"}
        )

        logging.info("OpenAPI reponse: {0}".format(response))

        response_choice = response["choices"][0]
        
        finish_reason = response_choice["finish_reason"]
        if finish_reason != "stop" and finish_reason != "function_call":
            logging.warn("GPT responded with unexpected finish reason: " +  finish_reason)

        response_message = response_choice["message"]
        if "function_call" in response_message:
            logging.info("GPT decided to call a function as expected")
            return response_message["function_call"]["arguments"]
        
        logging.warn("GPT did not infer function calling, best-effort answer parsing will be performed")
        return response_message["content"]

def get_new_initiative_details() -> str:
    prefixPattern = r'^\[.*-INITIATIVE\]:\s*' 

    initiative_details = f"""
    ### Initiative Display Name
    {re.sub(prefixPattern, "", os.environ["ISSUE_TITLE"], flags=re.IGNORECASE)}

    {os.environ["ISSUE_BODY"]}
    """
    
    # replace double-quotes with single quotes to minimize chance of invalid JSON returned by GPT
    return initiative_details.replace('"', "'")

if __name__ == "__main__":
    gpt_response = GPTAssistant().ask_gpt_for_initiative_json(get_new_initiative_details())
    
    respose_output_file = os.getenv("TEMP", "/tmp") + "/gpt-auto-comment.output"
    logging.info(f"Writing GPT Response to {respose_output_file}:\n{gpt_response}")

    with open(respose_output_file, "w", encoding="utf-8") as output:
        output.write(gpt_response)

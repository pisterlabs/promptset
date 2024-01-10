import os
import openai
import json

from .data_access import DataAccess

class CoachAgent:
    def __init__(self):
        self.data_access = DataAccess('aoe.db')
        
        self.key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = self.key

        self.model = 'gpt-3.5-turbo-0613'
        self.fnc = [
            {
                "name": "_get_civ_property",
                "description": "Get the selected property for a civilization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "civ": {
                            "type": "string",
                            "description": "The name of the civilization"
                        },
                        "property": {
                            "type": "string",
                            "description": "The name of the property. Either type, bonuses, unique_units, unique_techs, or team_bonus"
                        }
                    },
                    "required": ["civ", "property"]
                }
            },
            {
                "name": "check_unit_availability",
                "description": "Check whether a unit is available to a civilization",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "civ": {
                            "type": "string",
                            "description": "The name of the civilization"
                        },
                        "unit": {
                            "type": "string",
                            "description": "The name of the unit"
                        }
                    },
                    "required": ["civ", "unit"]
                }
            }
        ]
        self.available_functions = {
            "_get_civ_property": self.data_access._get_civ_property,
            "check_unit_availability": self.data_access.check_unit_availability
        }
        self.messages = [{"role": "system", "content": "You are an Age of Empires 2 chatbot that helps users learn and strategize." + 
                                                       " Keep answers concise and do not mention the game by name. Do not embellish the answer."},
                         {"role": "assistant", "content": "I understand. Awaiting user questions."}]

    def prompt(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def get_chat_completion(self):
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            functions=self.fnc,
            function_call="auto")

        return completion

    def process_chat_completion(self, completion):
        response_message = completion["choices"][0]["message"]

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_to_call = self.available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])


            if function_to_call.__name__ == "_get_civ_property":
                function_response = function_to_call(
                    civ=function_args.get("civ"),
                    property=function_args.get("property")
                )
            elif function_to_call.__name__ == "check_unit_availability":
                function_response = function_to_call(
                    civ=function_args.get("civ"),
                    unit=function_args.get("unit")
                )

            self.messages.append(response_message)
            self.messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )

            print(self.messages)
            second_response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
            )
            return second_response

        return None

    def handle_chat(self):
        completion = self.get_chat_completion()
        print(completion)
        second_response = self.process_chat_completion(completion)
        if second_response:
            print(second_response)



import os
import openai

class DataAgent:
    def __init__(self) -> None:
        self.key = os.environ.get('OPENAI_API_KEY')
        openai.api_key = self.key

        self.model = 'gpt-3.5-turbo-0613'

        self.fncs =  [ 
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
            }
        ]

    def get_data(self, statment: str) -> list[str]:

        self.messages = [
            {
                "role": "user", 
                "content": f"{statment}"
            } 
        ]

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            functions=self.fncs,
            function_call="auto"
        )

        print(completion)

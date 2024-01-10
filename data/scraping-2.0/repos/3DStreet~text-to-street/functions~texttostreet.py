import openai
import json
import os
import dotenv

config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']

#openai.api_key = os.getenv("OPENAI_API_KEY")

ASSIST_TEXT_FILENAME = 'prompt_bot.txt'

def load_prompt():
    with open(ASSIST_TEXT_FILENAME, "r") as file:
        return file.read()


def get_street(name, data):
    """Return streetmix JSON with street description"""
    # here will be valid JSON checking
    print(name)
    print(data)
    return json.dumps({"name": name, "data": data})


def get_streetmix_json(user_message):
    # Step 1: send the conversation and available functions to GPT
    assistant_description = load_prompt()
    messages = [
        {"role": "system", "content": assistant_description},
        {"role": "system", "content": "Only use the functions you have been provided with."},
        {"role": "user", "content": user_message}
    ]
    functions = [
        {
            "name": "get_street",
            "description": "get 3d street (url with 3dstreet) by its description",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "name of created street",
                        "default": "default street"
                    },
                    "data": {
                        "type": "object",
                        "properties": {
                            "rightBuildingVariant": {
                                "type": "string",
                                "description": "A string to determine which building variant to create for the right side of the street",
                                "enum": ["grass","narrow","residental","fence","parking-lot","waterfront","wide"],
                                "default": "grass"
                            },
                            "leftBuildingVariant": {
                                "type": "string",
                                "description": "A string to determine which building variant to create for the left side of the street",
                                "enum": ["grass", "narrow", "residental", "fence", "parking-lot", "waterfront", "wide"],
                                "default": "grass"
                            },
                            "street": {
                                "type": "object",
                                "properties": {
                                    "width": {
                                        "type": "number",
                                        "description": "street length in meters",
                                        "default": 150
                                    },
                                    "segments": {
                                        "type": "array",
                                        "description": "list of segments of a cross-section perspective of the 3D scene, each with a width in imperial feet units, a type in string format, and a variantString that applies modifications to the segment type",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "width": {
                                                    "type": "number",
                                                    "description": "segment width in imperial feet units",
                                                    "default": 9
                                                },
                                                "variantString": {
                                                    "type": "string",
                                                    "description": "Variant of segment. It's depend upon which segment type is selected. variantString values are separated by a pipe character (literally '|'). Most drive lane segments have an 'inbound' or 'outbound' value as the first variant."
                                                },
                                                "type": {
                                                    "type": "string",
                                                    "description": "street segment type",
                                                    "enum": ["sidewalk","streetcar","bus-lane","drive-lane","light-rail","streetcar","turn-lane","divider","temporary","stencils","food-truck","flex-zone","sidewalk-wayfinding","sidewalk-bench","sidewalk-bike-rack","magic-carpet","outdoor-dining","parklet","bikeshare","utilities","sidewalk-tree","sidewalk-lamp","transit-shelter","parking-lane"],
                                                    "default": "sidewalk"
                                                },
                                                "elevation": {
                                                    "type": "number",
                                                    "description": "elevation level for segment. 1 is default for all type of sidewalks and buildings, 0 is default for roads, lanes, parking, etc",
                                                    "enum": [0, 1],
                                                    "default": 1
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                },
                "required": ["data"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
        # auto is default. "auto" means the model can pick between an end-user or calling a function
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_street": get_street,
        }
        function_name = response_message["function_call"]["name"]
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        print(function_args)
        function_response = fuction_to_call(
            name=function_args.get("name"),
            data=function_args.get("data"),
        )

        return function_response # return JSON from get_street

    return response_message # in case if GPT not called a function

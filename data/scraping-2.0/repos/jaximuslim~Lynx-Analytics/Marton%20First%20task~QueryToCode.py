import pandas as pd
import openai
import requests
import json
from dotenv import dotenv_values

# import keys from .env file
env_values = dotenv_values(".env")
openai.api_key = env_values["OPENAI_API_KEY"]
OPENCAGE_API_KEY = env_values["OPENCAGE_API_KEY"]
openai.organization = "org-1gDv2KDgsLFDTtbzcWRU7TvP"

csv_file_path = (
    "C:\\Users\\User\\Tutorial\\Marton First task\\person_details - person_details.csv"
)


def get_location_info(city_name):
    """This function takes the city a person is in if the city is in the United States and checks if the city is within a state"""
    url = f"https://api.opencagedata.com/geocode/v1/json?q={city_name}&key={OPENCAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    city_info = data["results"][0]["components"]
    return json.dumps(city_info)


def get_csv_file_as_string(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df.to_string(index=False)


def interpret_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df


def auto_filter():
    csv_file = get_csv_file_as_string(csv_file_path)

    delimiter = "####"
    instruction = f"""
    You are a chatbot that can read dataframes and answer questions about them.\
    The follwing information is how you should interpret the dataframe. \
    Each row contains details about a person. \
    The first column is the names of the person in the dataframe. \
    The second column is the age of the person in integer format. \ 
    The third column represents the city the person lives in. \
    The fourth column represents the country the person lives in. \
    
    """

    messages = [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": "for this dataframe,"
            + csv_file
            + "who lives in the state of California?",
        },
    ]
    functions = [
        # {
        #     "name": "interpret_csv_file",
        #     "description": "Interpret the csv file and returns a dataframe",
        #     "parameters": {"type": "Dataframe", "properties": {}},
        # },
        {
            "name": "get_location_info",
            "description": "Get state city is in",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The name of the city",
                    },
                    "continent": {
                        "type": "string",
                        "description": "The continent the city is located in",
                    },
                    "country": {
                        "type": "string",
                        "description": "The country the city is located in",
                    },
                    "country_code": {
                        "type": "string",
                        "description": "The country code of the country the city is located in",
                        "pattern": "[A-Z]{2}",
                    },
                    "state": {
                        "type": "string",
                        "description": "The state the city is located in",
                    },
                },
                "required:": ["city"],
            },
        },
    ]

    api_resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
        max_tokens=512,
    )

    answer = api_resp["choices"][0]["message"]
    print(answer)

    if answer.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        print("ok")
        available_functions = {
            "get_location_info": get_location_info,
        }
        function_name = answer["function_call"]["name"]
        print(answer)
        print(function_name)
        fuction_to_call = available_functions[function_name]
        function_args = json.loads(answer["function_call"]["arguments"])
        function_response = fuction_to_call(
            city=function_args.get("location"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(answer)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=0.0,
            top_p=1,
            max_tokens=512,
        )  # get a new response from GPT where it can see the function response
        # print(second_response)
        return second_response


auto_filter()

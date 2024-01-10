import openai
import api_key
import os
import json


def Extract_loc(msg):
    openai.api_key = api_key.API_GPT_Jake

    # schema to fix the strucuture of GPT reply
    places_data_schema = {
        "type" : "object",
        "properties": 
        {
            "departure": 
            {
                "type": "string",
                "description": "The departure place."
            },
            "arrival": 
            {
                "type": "string",
                "description": "The arrival place."
            }
        },
        "required": ["departure", "arrival"]
    }

    # parameters to set GPT Chatbot 
    parameters = {"temperature": 0.2,
                  "top_p": 0.2, 
                  "max_tokens": 100
                  }

    # create chatbot
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "{출발지, 목적지} 두 단어만 출력해줘. 출발지나 목적지라는 단어는 출력하지 말고, 두 단어만 쉼표로 구분해서 적어줘"},
            {"role": "assistant", "content": "{출발지, 목적지} 두 단어만 출력해줘. 출발지나 목적지라는 단어는 출력하지 말고, 두 단어만 쉼표로 구분해서 적어줘"},
            {"role": "user", "content": msg},
        ],
        # Implement a function call with JSON output schema
        functions=[{
        "name": "get_departure_arrival_name",
        "description": "Get departure and arrival words about the given sentence. There are only two correct words.",
        "parameters": places_data_schema
        }],
            
        # Define the function which needs to be called when the output has received
        function_call = {
            "name" : "get_departure_arrival_name"
        },
        ## parameters for calling GPT chatbot
        temperature=parameters["temperature"],
        top_p=parameters["top_p"],
        max_tokens=parameters["max_tokens"]
    )

    # 문자열을 JSON으로 변환
    gpt_answer = response["choices"][0]["message"]["function_call"]["arguments"]
    result_json = json.loads(gpt_answer)

    # get departure and arrival information
    departure = result_json["departure"]
    arrival = result_json["arrival"]
    return [departure, arrival]

# print(Extract_loc('군위구청 동대구역 가는길 알려줘'))

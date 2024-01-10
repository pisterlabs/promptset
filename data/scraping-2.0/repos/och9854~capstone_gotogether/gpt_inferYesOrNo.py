import openai
import api_key
import os
import json
import re


def Yes_or_No(msg):
    msg = re.sub("고마워", "", msg, flags=re.IGNORECASE)
    openai.api_key = api_key.API_GPT_Minse

    # schema to fix the strucuture of GPT reply
    answer_data_schema = {
        "type" : "object",
        "properties": 
        {
            "answer": 
            {
                "type": "string",
                "description": "{Yes, No} 중 하나만 출력해줘."
            }
        },
        "required": ["answer"]
    }

    # parameters to set GPT Chatbot 
    parameters = {"temperature": 0.3,
                  "top_p": 0.3, 
                  "max_tokens": 30
                  }

    # create chatbot
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "{Yes, No} 중 하나만 출력해줘. "},
            {"role": "assistant", "content": "{Yes, No} 중 하나만 출력해줘. "},
            {"role": "user", "content": msg},
        ],
        # Implement a function call with JSON output schema
        functions=[{
        "name": "get_answer",
         "description": "{Yes, No} 중 하나만 출력해줘.",
        "parameters": answer_data_schema
        }],
            
        # Define the function which needs to be called when the output has received
        function_call = {
            "name" : "get_answer"
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
    return result_json["answer"]



# print(Yes_or_No('응'))

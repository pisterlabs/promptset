import json
from bardapi import Bard
from datetime import datetime
import time
import requests


def _bardai(json_data):
    
    queryString = json_data["prime1"]
    ##Secure-1PSID
    token = '[Secure-1PSID]'
    bard = Bard(token=token)

    out = bard.get_answer(queryString)['content']
    json_data["content1"] = out
    ##json_data["prime1"] = ""

    return json_data

def _openai(json_data):

    url = "https://api.openai.com/v1/chat/completions"
        
    payload_json = {
    "model": "gpt-4",
    "messages": [
        {
        "role": "system",
        "content": "You are University advisor at UBC, verify the following information and offer additional suggestions for advanced studies:"
        },
        {
        "role": "user",
        "content": ""
        }
    ]}
    payload_json["messages"][1]["content"] = json_data["prime1"]
    payload = json.dumps(payload_json)

    headers = {
    'Content-Type': 'application/json',
    'Authorization': '[OpenAI API Key]'
    }

    ##remove request for now, since Autho API key is no longer valid.  
    ##response = requests.request("POST", url, headers=headers, data=payload)
    ##json_data["content1"] = response["choices"][0]["message"]["content"]
    print(json.dumps(payload_json, indent=2))


    return json_data

if __name__ == "__main__":
    bard_json_data = {
        "prime1": "value1",
        "content1": "value2",
        "key3": ["item1", "item2", "item3"]
    }
    openai_json_data = {
        "prime1": "value1",
        "content1": "value2",
        "key3": ["item1", "item2", "item3"]
    }

for i in range(3):
    print(f"\nIteration {i + 1}:")
    # Call function A with JSON data and retrieve the result
    if i == 0:
        bard_json_data["prime1"] = "PRIME WITH **Core courses for business*** COMM 101 - Business Fundamentals: grade above 90 then move on to COMM 291; grade between 85 and 90 student can move to COMM 291A; grade between 70 and 85 student can move to COMM 291B, else repeat COMM 101* COMM 291 - Application of Statistics in Business: grade above 90 then move on to COMM 292; grade between 85 and 90 student can move to COMM 292A; grade between 70 and 85 student can move to COMM 292B, else repeat COMM 291* COMM 291A - Application of Statistics in Business* COMM 291B - Application of Statistics in Business* COMM 292 - Management and Organizational Behaviour: grade above 70 then move on to ECON 101 or MATH 104; else repeat COMM 291* COMM 292A - Application of Statistics in Business * COMM 292B - Application of Statistics in Business * ECON 101 - Principles of Microeconomics * MATH 104 - Differential Calculus with Applications to Commerce and Social Sciences * OVERALL average of COMM 101, COMM 291 and COMM 292 between 75 and 90, the student can only take MINOR of ECON * OVERALL average of COMM 101, COMM 291 and COMM 292 above 90, the student can move to a quantitative MAJOR in sauder such as Accounting, Finance, BTM, Supply chain and logistics"
    result_from_BardAI = _bardai(bard_json_data)
    print(f"Result from BardAI[{i+1}]:")
    print(json.dumps(result_from_BardAI, indent=2))

    time.sleep(5)

    # Call OpenAI with JSON data and retrieve the result
    openai_json_data["system"] = "You are University advisor at UBC, verify the following information and offer additional suggestions for advanced studies"
    if i == 0:
        openai_json_data["prime1"] = bard_json_data["prime1"]
    else:    
        openai_json_data["prime1"] = result_from_BardAI["content1"]
    result_from_OpenAI = _openai(openai_json_data)
    print(f"Result from OpenAI[{i+1}]:")
    print(json.dumps(result_from_OpenAI, indent=2))

    ##Send back to BARD AI for further assessment.
    ##bard_json_data["prime1"] = "You are University advisor at UBC, verify the following information and offer additional suggestions for advanced studies: '" + result_from_OpenAI["content1"] + "'";
    bard_json_data["prime1"] = "You are University advisor at UBC, verify the following information and offer additional suggestions for advanced studies: '" + result_from_OpenAI["prime1"] + "'";


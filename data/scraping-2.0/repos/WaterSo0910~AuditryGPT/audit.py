import os
from type import Result
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def find_vulnerabilities(prompt):
    prompt = prompt.replace("    ", "").replace("\t", "") + "\n\nThe vulnerability is:"
    response = openai.Completion.create( 
        model=os.getenv("MODEL_NAME"), 
        prompt=prompt, 
        temperature=0, 
        max_tokens=1024, 
        top_p=1, 
        frequency_penalty=0.5, 
        presence_penalty=0, stop=["\n", " User:", " AI:"] 
    )
    return response["choices"][0]["text"]

def parse_solidity(content):
    paranthesis = 0
    current_contract = []
    contract_segments = []
    for i in content.split("\n"):
        if "{" in i:
            paranthesis += 1
        if "}" in i:
            paranthesis -= 1
        if paranthesis != 0:
            current_contract.append(i)
            if len(current_contract) > 5:
                contract_segments.append("\n".join(current_contract))
                current_contract = []
    if len(current_contract)>1:
        contract_segments.append("\n".join(current_contract))
    return contract_segments

def audit(contract: str)-> list[Result]:
    Results:list[Result] = []
    parsed = parse_solidity(contract)
    for i in parsed:
        Results.append(Result(function=i, audit="🤖️ " + find_vulnerabilities(i)))
    return Results
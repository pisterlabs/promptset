import os
import json

from langchain.llms import AzureOpenAI

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt3

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2022-12-01" #"2023-05-15"
os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

def inventory_api_extract_info(input: str) -> str:
#    prompt = f"Extract `tire` and `store` information from `{input}` and create a json object. If the information is missing, leave it empty. Example: `I want to buy Michelin tires from store Issaquah` becomes {'store': 'Issaquah', 'tire': 'Michelin'}. Example: `I want to buy tires at my local store Bellevue` becomes {'store': 'Bellevue', 'tire': ''}. Example: `I want to buy four goodyear tires` becomes {'store': '', 'tire': 'goodyear'}. "
    prompt = 'Extract `tire` and `store` information from ' + input + ' and create a json object. If the information is missing, leave it empty. Example: I want to buy Michelin tires from store Issaquah becomes {"store": "Issaquah", "tire": "Michelin"}. Example: I want to buy tires at my local store Bellevue becomes {"store": "Bellevue", "tire": ""}. Example: I want to buy four goodyear tires becomes {"store": "", "tire": "goodyear"}. '

    print(f"prompt: {prompt}")
    # Create an instance of Azure OpenAI
    # Replace the deployment name with your own
    llm = AzureOpenAI(
        deployment_name=Az_Open_Deployment_name_gpt3,
        model_name="text-davinci-003", 
    )

    result = llm(prompt)

    print(f"inventory_api_extract_info: {result}")

    return result

def inventory_api_json(input: str):
    obj = json.loads(input)

    return obj


def inventory_api_v2(input: str) -> str:
    """Searches the inventory information for `tire` in `store`. The requied parameter `input` is text in message of agent run."""

    input_str = inventory_api_extract_info(input)

    info = inventory_api_json(input_str)

    if(info['tire'] and info['store']):
        
        return f"There are 10 {info['tire']} available in store {info['store']}."
    elif (info['tire']):
        return "Please ask human to provide `store` information"
    else:
        return "Please ask human to provide `tire` information"

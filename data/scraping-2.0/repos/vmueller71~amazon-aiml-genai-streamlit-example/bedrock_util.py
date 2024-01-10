import boto3
import botocore
import constants as const
from langchain.llms.bedrock import Bedrock

region = const.region_name
url = const.bedrock_ep_url

inference_modifier_titan = { 
    "maxTokenCount": 4096,
    "stopSequences": [],
    "temperature":0.1,  
    "topP":0.9
}

inference_modifier_claude = {
    'max_tokens_to_sample':4096,
    "temperature":0.5,
    "top_k":250,
    "top_p":1,
    "stop_sequences": ["\n\nHuman"]
}

def bedrock_test():
    bedrock= boto3.client(service_name='bedrock', region_name=region, endpoint_url=url)
    output_text = bedrock.list_foundation_models()
    return output_text
    
def get_genAI_llm():
    session = boto3.Session()
    bedrock= session.client(service_name='bedrock', region_name=region, endpoint_url=url)
    genAI_llm = Bedrock(model_id = "amazon.titan-tg1-large",
        client = bedrock,
        model_kwargs = inference_modifier_titan)
    return genAI_llm
import os
import sys
import boto3
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

def bedrock_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    inference_modifier = {
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": stop_sequence,
    }

    textgen_llm = Bedrock(
        model_id=model_id,
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )

    return textgen_llm(prompt)

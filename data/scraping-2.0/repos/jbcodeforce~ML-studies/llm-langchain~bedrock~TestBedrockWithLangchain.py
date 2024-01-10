import json
import os
import sys
import botocore
from langchain.llms import Bedrock

module_path = "."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

bedrock_runtime = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
)

inference_modifier = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

'''
`model_id` of the model available in Amazon Bedrock. 

Optionally we can pass on a previously created boto3 `client` as well as 
some `model_kwargs` which can hold parameters such as `temperature`, `topP`, `maxTokenCount` 
or `stopSequences` (more on parameters can be explored in Amazon Bedrock console).
'''
textgen_llm = Bedrock(
    model_id="anthropic.claude-v2",
    client=bedrock_runtime,
    model_kwargs=inference_modifier,
)

response = textgen_llm("""

Human: Write an email from Bob, Customer Service Manager, 
to the customer "John Doe" that provided negative feedback on the service 
provided by our customer support engineer.

Assistant:""")

print_ww(response)


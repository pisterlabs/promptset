import json
import boto3
import os
import time
from multiprocessing import Process
from io import BytesIO
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

endpoint_name = os.environ.get('endpoint')
    
# initiate llm model based on langchain
class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({'inputs': prompt, 'parameters': model_kwargs})
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"]

content_handler = ContentHandler()

aws_region = boto3.Session().region_name

parameters = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "do_sample": False,
    "temperature": 0.5,
    "repetition_penalty": 1.03,
    "top_p": 0.9,
    "top_k":1,
    "stop": ["<|endoftext|>", "</s>"]
}      
        
llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = aws_region, 
    model_kwargs = parameters,
    content_handler = content_handler
)
    
def lambda_handler(event, context):
    print(event)

    text = event['text']
    print('text: ', text)

    start = int(time.time())

    generated_text = llm(text)
    print('generated_text: ', generated_text)

    if generated_text == '':
        generated_text = 'Fail to read the document. Try agan...'

    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)

    return {
        'statusCode': 200,
        'msg': generated_text,
    }

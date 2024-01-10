import json
import boto3
import os
import time
from io import BytesIO
import PyPDF2
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
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

def get_summary_from_pdf(file_type, s3_file_name):
    summary = ''
    
    if file_type == 'pdf':
        s3r = boto3.resource("s3")
        doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
        
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    

        new_contents = str(contents).replace("\n"," ") 
        print('length: ', len(new_contents))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        texts = text_splitter.split_text(new_contents) 
        print('texts[0]: ', texts[0])
        
        docs = [
            Document(
                page_content=t
            ) for t in texts[:3]
        ]
        prompt_template = """Write a concise summary of the following:

        {text}
        
        CONCISE SUMMARY """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        summary = chain.run(docs)
                    
    if summary == '':  # error notification
            summary = 'Fail to summarize the document. Try agan...'
    print('summary: ', summary)

    return summary[1:len(summary)-1]
        
def lambda_handler(event, context):
    print(event)

    object = event['object']
    print('object: ', object)
    
    start = int(time.time())
    
    summary = get_summary_from_pdf('pdf', object)

    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)
    
    return {
        'statusCode': 200,
        'msg': summary,
    }                                       
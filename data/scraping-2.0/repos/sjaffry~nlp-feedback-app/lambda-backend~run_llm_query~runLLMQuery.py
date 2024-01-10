from langchain.embeddings import BedrockEmbeddings
from langchain.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
import os
import boto3
import json
import base64

def decode_base64_url(data):
    """Add padding to the input and decode base64 url"""
    missing_padding = len(data) % 4
    if missing_padding:
        data += '=' * (4 - missing_padding)
    return base64.urlsafe_b64decode(data)

def decode_jwt(token):
    """Split the token and decode each part"""
    parts = token.split('.')
    if len(parts) != 3:  # a valid JWT has 3 parts
        raise ValueError('Token is not valid')

    header = decode_base64_url(parts[0])
    payload = decode_base64_url(parts[1])
    signature = decode_base64_url(parts[2])

    return json.loads(payload)
    #return { 'business_name': payload['cognito:groups'][0] }
    
def trimmed_foldername(full_folderpath):
    return os.path.basename(os.path.normpath(full_folderpath))

def download_index_files(bucket_name, index_file, pkl_file, local_index_file, local_pkl_file):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, index_file, local_index_file)
        s3.download_file(bucket_name, pkl_file, local_pkl_file)
        print("Files downloaded successfully.")
    except Exception as e:
        raise ValueError(e)

def lambda_handler(event, context):

    bucket_name = os.environ['bucket_name']
    date_range = event["queryStringParameters"]['date_range']
    query = event["queryStringParameters"]['query']
    prompt_prefix = "Respond in English only. "
    
    # Let's extract the business name from the token by looking at the group memebership of the user
    token = event['headers']['Authorization']
    decoded = decode_jwt(token)
    # We only ever expect the user to be in one group only - business rule
    business_name = decoded['cognito:groups'][0]

    # Download faiss index files if not already exist
    index_loc=f"transcribe-output/{business_name}/{date_range}/faiss_index"
    index_file = f"{index_loc}/index.faiss"
    pkl_file = f"{index_loc}/index.pkl"
    local_path_prefix = f"/tmp/{date_range}"
    local_path = f'{local_path_prefix}/faiss_index_{business_name}'
    local_index_file = f"{local_path}/index.faiss"
    local_pkl_file = f"{local_path}/index.pkl"

    if not(os.path.isfile(local_index_file) and os.path.isfile(local_pkl_file)):
            print("Files don't exist")
            os.mkdir(local_path_prefix)
            os.mkdir(local_path)
            download_index_files(bucket_name, index_file, pkl_file, local_index_file, local_pkl_file)
    else:
        print("Files already exists no need to download")

    #Bedrock client
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )

    embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_runtime,
    region_name="us-east-1",
    )

    # Load faiss index
    docsearch = FAISS.load_local(local_path, embeddings)
    model_kwargs = {
    'max_tokens_to_sample': 1000,
    'temperature': 1.0
    }

    llm = Bedrock(
        model_id="anthropic.claude-v2", model_kwargs=model_kwargs, client=bedrock_runtime, region_name="us-east-1"
    )
    memory = ConversationSummaryMemory(
        llm=llm, memory_key="chat_history", return_messages=True
    )

    # Initialize the retrieval chain and run LLM query
    try:
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever(), memory=memory)
        result = qa.run(prompt_prefix+query)
    except Exception as e:
        raise ValueError(e)

    return {
            'statusCode': 200,
            'headers': {
                "Access-Control-Allow-Headers" : "Content-Type",
                "Access-Control-Allow-Origin": "https://query.shoutavouch.com",
                "Access-Control-Allow-Methods": "OPTIONS,PUT,POST,GET"
        },    
            'body': json.dumps(result)
        }    
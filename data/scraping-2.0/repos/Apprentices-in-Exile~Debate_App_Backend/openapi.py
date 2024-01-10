import json

import os
import openai

import boto3
from botocore.exceptions import ClientError

client = boto3.client('apigatewaymanagementapi', endpoint_url="https://a0ppckpw77.execute-api.us-east-2.amazonaws.com/development")


def get_secret():

    secret_name = "openai_api_key"
    region_name = "us-east-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    api_key_dict = json.loads(secret)
    os.environ['OPENAI_API_KEY'] = api_key_dict["openai_api_key"]
    openai.api_key = api_key_dict["openai_api_key"]
    #print('test ', os.environ['OPENAI_API_KEY'][:5]) 


def lambda_handler(event, context):
    get_secret()
    print("=-=-=-===", event)
    #Extract connectionId from incoming event
    connectionId = event["requestContext"]["connectionId"]
    
    body = json.loads(event["body"])
    message = body["message"]
    num_rebuttals = 2
    model = "gpt-3.5-turbo"
    prompt = f"""
    Argue the topic of {message[0]} from both the perspective {message[1]} and {message[2]}. Go back and forth for {num_rebuttals} iterations.
    """
    messages = [
        {"role": "user", "content": f"{prompt}"}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True
    )
    
    for index, chunk in enumerate(response):
        if "content" in chunk["choices"][0]["delta"]:
            client.post_to_connection(ConnectionId=connectionId, Data=chunk["choices"][0]["delta"]["content"])
    return { "statusCode": 200  }

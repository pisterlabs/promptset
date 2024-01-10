import boto3
import base64
from botocore.exceptions import ClientError
import json
import openai

openai.api_key = get_secret()

def lambda_handler(event, context):
    text_ingress = event['body']
    resp = ai_function(text_ingress)
    return {
        'statusCode': 200,
        'body': json.dumps(resp)
    }

def ai_function(text_function):
    response = openai.Completion.create(
      model="text-davinci-002",
      prompt=text_function,
      temperature=0,
      max_tokens=260,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      stop=["**"]
    )
    print(response)
    text_out = response["choices"][0]["text"]
    
    return text_out


def get_secret():

    secret_name = "openai_secret"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            secretn= base64.b64decode(get_secret_value_response['SecretBinary'])
            
    return secret

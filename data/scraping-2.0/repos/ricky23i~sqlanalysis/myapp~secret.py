
import json
import openai
import boto3
from botocore.exceptions import ClientError

secrets_manager_client = boto3.client('secretsmanager', region_name='us-east-1')

def get_secret(key_name):
    secret_name = key_name

    try:
        response = secrets_manager_client.get_secret_value(SecretId=secret_name)
        secret_data = json.loads(response['SecretString'])
        return secret_data[secret_name]

    except ClientError as e:
        print(f"Error retrieving OpenAI API key from AWS Secrets Manager: {e}")
        raise



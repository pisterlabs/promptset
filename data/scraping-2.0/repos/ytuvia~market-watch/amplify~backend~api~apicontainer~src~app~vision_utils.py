from openai import OpenAI
import json
import os
import time
import boto3

def get_secret(secret_name):
    region_name = os.environ.get('REGION');

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(
        SecretId=secret_name
    )

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    return secret

api_key = get_secret('OPENAI_API_KEY')
GPT_MODEL="gpt-4-vision-preview"
client = OpenAI(
    api_key=api_key 
)

def image_question(url, question):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                },
            ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content
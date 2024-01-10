import os
import json
import traceback
import openai
import boto3
from typing import Any, Dict
from dotenv import dotenv_values

config = {
    **dotenv_values(os.path.join(os.path.dirname(__file__), '..', '.env')),
    **os.environ,
}

openai.api_key = config.get('OPENAI_API_KEY')
BUCKET_NAME = config.get('BUCKET_NAME')


def handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    try:
        uid = event.get('requestContext', {}).get(
            'authorizer', {}).get('principalId')

        body = event.get('body')

        if body is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': 'please provide POST body'}),
                'headers': {
                    'Content-Type': 'application/json',
                },
            }

        body = json.loads(body)
        key = body.get('key')

        s3_resource = boto3.resource('s3')

        s3_object = s3_resource.Object(BUCKET_NAME, key)
        s3_response = s3_object.get()
        file_response = openai.File.create(file=s3_response['Body'], purpose='fine-tune')
        file_id = file_response.get('id', '')
        finetune_response = openai.FineTune.create(training_file=file_id)
        finetune_job_id = finetune_response.get('id', file_id)
        new_s3_object = s3_resource.Object(BUCKET_NAME, f'{uid}/{finetune_job_id}')
        new_s3_object.copy_from(CopySource=dict(Bucket=BUCKET_NAME, Key=key))
        s3_object.delete()

        return {
            'statusCode': 200,
            'body': json.dumps(finetune_response),
            'headers': {
                'Content-Type': 'application/json',
            },
        }
    except Exception:
        traceback.print_exc()
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'error occurred, please check logs'}),
            'headers': {
                'Content-Type': 'application/json',
            },
        }


if __name__ == '__main__':
    print(config)

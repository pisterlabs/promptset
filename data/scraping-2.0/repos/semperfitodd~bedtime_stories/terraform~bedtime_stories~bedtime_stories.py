import os
import openai
import boto3
import json
from datetime import datetime
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION_NAME = boto3.session.Session().region_name

def get_secret(secret_name):
    client = boto3.client(service_name='secretsmanager')
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except Exception as e:
        raise e
    else:
        secret = json.loads(get_secret_value_response['SecretString'])
        return secret

def generate_story(characters, subject):
    characters_list = ', '.join(characters)
    prompt = f"Please create a child-friendly {subject} story featuring the characters: {characters_list}. Make it engaging and suitable for a 10-year-old reader. It should be around 1000 words."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1500,
        temperature=0.7
    )
    return response.choices[0].text

def store_story(date, characters, subject, story_response, table):
    item = {
        'date': date,
        'characters': characters,
        'subject': subject,
        'story': story_response
    }
    table.put_item(Item=item)

def synthesize_audio(story_text):
    polly_client = boto3.client('polly', region_name=REGION_NAME)
    response = polly_client.synthesize_speech(
        OutputFormat='mp3',
        Text=story_text,
        VoiceId='Joanna'
    )
    return response['AudioStream'].read()

def generate_presigned_url(bucket_name, key):
    s3_client = boto3.client('s3', region_name=REGION_NAME)
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': key},
        ExpiresIn=172800  # 2 days in seconds
    )
    return url

def store_audio_in_s3(audio_stream, key):
    bucket_name = os.getenv('POLLY_BUCKET')
    s3_client = boto3.client('s3', region_name=REGION_NAME)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=audio_stream
    )
    # Generate pre-signed URL
    presigned_url = generate_presigned_url(bucket_name, key)
    return presigned_url
def lambda_handler(event, context):
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
    }

    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers
        }

    environment = os.getenv("ENVIRONMENT")
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(environment)

    if event['httpMethod'] == 'POST':
        try:
            body = json.loads(event.get('body', '{}'))
            characters = body.get('characters')
            subject = body.get('subject')

            if not characters or not subject:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'message': 'Characters and subject are required'})
                }

            secret_name = f"{environment}_secret"
            secrets = get_secret(secret_name)
            openai.organization = secrets['openai_org']
            openai.api_key = secrets['openai_key']

            generated_story = generate_story(characters, subject)
            # Synthesize the audio
            audio_stream = synthesize_audio(generated_story)

            # Define the S3 key for the audio file
            audio_key = f"stories/{context.aws_request_id}.mp3"

            # Store the audio in S3
            store_audio_in_s3(audio_stream, audio_key)

            current_date_string = datetime.now().strftime('%Y-%m-%d')
            store_story(current_date_string, characters, subject, generated_story, table)

            presigned_url = store_audio_in_s3(audio_stream, audio_key)

            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({'story': generated_story, 'url': presigned_url,
                                    'message': 'Story generated and stored successfully'})
            }

        except Exception as e:
            logger.error("Error processing request: {}".format(e))
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'message': 'Internal Server Error'})
            }

    else:
        return {
            'statusCode': 400,
            'headers': cors_headers,
            'body': json.dumps({'message': 'Invalid request method'})
        }

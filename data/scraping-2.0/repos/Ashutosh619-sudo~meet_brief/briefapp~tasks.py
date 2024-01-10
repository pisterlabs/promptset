from .models import Caption, MeetingSummary
import boto3
import botocore
import os
import openai
import json

openai.api_key = os.environ['OPEN_AI_KEY']

def _setup_boto():

    aws_key = os.environ['AWS_ACCESS_KEY']
    aws_secret = os.environ['AWS_SECRET_KEY']

    session = boto3.Session(aws_access_key_id=aws_key,aws_secret_access_key=aws_secret)
    s3 = session.client('s3')
    return s3

def get_summary(data):

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Summarize this: {data}"},]
    )

    summary = response["choices"][0]["message"]["content"]

    return summary


def summarize_caption(caption_id):

    caption = Caption.objects.filter(id=caption_id)
    s3 = _setup_boto()
    bucket_name = os.environ['AWS_BUCKET']
    object_key = caption[0].object_key
    file_name = caption[0].file_name


    try:
        with open(file_name, 'wb') as f:
            s3.download_fileobj(bucket_name, object_key, f)

        f = open(file_name)
        data = json.load(f)
        print(data)

        print(get_summary(data['conversation']))

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

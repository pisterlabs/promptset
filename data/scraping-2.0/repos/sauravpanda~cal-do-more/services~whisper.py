from openai import OpenAI
import os
import boto3


def whisper_video(bucket_name, object_key):
    # Access the variables
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Set your AWS credentials (replace 'your_access_key' and 'your_secret_key' with your actual credentials)
    aws_access_key = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_key = os.environ.get("AWS_SECRET_KEY")

    s3 = boto3.client(
        "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
    )

    # response = s3.list_objects_v2(Bucket=bucket_name)

    local_path = f"audio-folder/{object_key}"

    s3.download_file(bucket_name, object_key, local_path)
    client = OpenAI(api_key=openai_api_key)

    audio_file = open(f"audio-folder/{object_key}", "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    return transcript.text

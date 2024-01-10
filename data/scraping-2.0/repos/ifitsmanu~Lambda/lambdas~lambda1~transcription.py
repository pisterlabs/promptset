# import all required modules
import json
import os
from deepgram import Deepgram
import openai
import boto3
import logging
from urllib.request import urlopen
import io
import asyncio

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_presigned_url(s3_url, expiration=3600):
    """
    Generate a presigned URL for an object in an S3 bucket.
    """
    try:
        s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
        bucket_name = s3_url.split("//")[1].split(".")[0]
        object_key = s3_url.split(bucket_name + ".s3.amazonaws.com/")[1]
        response = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": object_key},
            ExpiresIn=expiration,
        )
    except Exception as e:
        print(e)
        return None
    return response


async def deepgram_transcribe(s3_signed_url):
    # create deepgram client
    dg_client = Deepgram(os.environ.get("DEEPGRAM_API_KEY"))
    options = {"punctuate": True, "model": "general", "tier": "enhanced"}
    source = {"url": s3_signed_url}
    response = await dg_client.transcription.prerecorded(source, options)
    transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcription


def openai_transcribe(s3_signed_url, filename):
    # get whisper api key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    with urlopen(s3_signed_url) as response:
        audio = response.read()
        audio_file = io.BytesIO(audio)
        # get audio file name from s3 url
        audio_file.name = filename
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        transcription = transcript.text
        return transcription


def lambda_handler(event, context):
    body = json.loads(event.get("body", "{}"))
    logger.info(f"Body: {body}")
    # get s3 audio url from body
    s3_audio_url = body.get("s3_audio_url", "")
    logger.info(f"s3_audio_url: {s3_audio_url}")
    # check if s3 audio url ends with .wav
    if not s3_audio_url.endswith(".wav"):
        return {"statusCode": 400, "body": json.dumps("Audio file must be .wav")}
    # get name of api to use to generate transcription.
    api = body.get("api_name", "whisper")
    logger.info(f"api: {api}")
    # get signed url
    s3_signed_url = get_presigned_url(s3_audio_url)
    logger.info(f"signed_url: {s3_signed_url}")
    if api == "deepgram":
        # get deepgram transcription
        transcription = asyncio.run(deepgram_transcribe(s3_signed_url))
    elif api == "whisper":
        transcription = openai_transcribe(s3_signed_url, s3_audio_url.split("/")[-1])
    else:
        return {"statusCode": 400, "body": json.dumps("Invalid api name")}
    logger.info(f"Transcription: {transcription}")
    return {"statusCode": 200, "body": json.dumps(transcription)}

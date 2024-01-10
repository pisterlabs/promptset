import os
import json
import openai
import boto3
import pandas as pd
import asyncio
from dotenv import load_dotenv
from datetime import datetime


# ------------------Async------------------#
async def process_audio_async(filename, obj_size, openai_key):
    """
    Converts an audio file to text using OpenAI's Audio.transcribe method
    :param filename: The name of the file that is being processed.
    :param obj_size: The size of the audio file in MB.
    :param openai_key: The OpenAI API key.
    :return: The text that is being appended to the JSONL file.
    """
    openai.api_key = openai_key
    with open(data_path + filename, "rb") as f:
        result = openai.Audio.translate("whisper-1", f)
    transcript = result["text"]
    os.remove(data_path + filename)

    return transcript


async def process_audio(loop, obj, data_path, openai_keys):
    filename = obj.key.split('/')[-1]
    if os.path.exists(data_path + 'data.jsonl'):
        with open(data_path + 'data.jsonl', 'r') as f:
            if filename in f.read():
                print(f"{datetime.now().replace(microsecond=0)} ({obj.size / 1e6:.2f} MB) already "
                      f"processed {filename}")
                return None
    print(f"{datetime.now().replace(microsecond=0)} ({obj.size / 1e6:.2f} MB) working on {filename}")

    if (filename == "") or (0.1 > (obj.size / 1e6) or (obj.size / 1e6) > 25):
        return None
    else:
        mp3_content = obj.get()['Body'].read()
        with open(data_path + filename, 'wb') as f:
            f.write(mp3_content)
        obj_date = obj.last_modified
        obj_size = f"{obj.size / 1e6:.2f}"
        openai_key = openai_keys[obj.key % len(openai_keys)]
        start = datetime.now()
        transcript = await process_audio_async(filename, obj_size, openai_key)
        end = datetime.now()
        tokens = (end - start).seconds * 200
        cost = f"{(end - start).seconds * 0.006:.3f}"
        return [filename, transcript, tokens, obj_date, (end - start).seconds, cost, obj_size]


async def main(loop, config, openai_keys):
    bucket, bucket_path, bucket2, data_path = config

    tasks = []
    semaphore = asyncio.Semaphore(5)
    async with semaphore:
        for obj in bucket.objects.filter(Prefix=bucket_path):
            task = loop.create_task(process_audio(loop, obj, data_path, openai_keys))
            tasks.append(task)
        results = await asyncio.gather(*tasks)

    data = [result for result in results if result is not None]

    df = pd.DataFrame(data, columns=['filename', 'transcript', 'tokens', 'recording_date', 'time', 'cost', 'size'])
    df['transcription_date'] = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    df['status'] = 'pending'

    df.to_json(data_path + 'data2.jsonl', orient='records', lines=True)


# ------------------Main------------------#
if __name__ == '__main__':
    load_dotenv()
    openai_keys = os.getenv("OPENAI_API_KEYS").split(',')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-west-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    bucket = s3.Bucket('recordings3cx')
    bucket_path = 'RU/7520/'
    bucket2 = s3.Bucket('3cxtranscriptions')
    data_path = '../data/'

    config = (bucket, bucket_path, bucket2, data_path)

    asyncio.run(main(asyncio.get_event_loop(), config, openai_keys))

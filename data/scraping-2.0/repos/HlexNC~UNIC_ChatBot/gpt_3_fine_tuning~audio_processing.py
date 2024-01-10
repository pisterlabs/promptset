"""
:author: Alex Rudaev
:created_on: 2023-02-14
:updated_on: 2023-03-07
:description: This script is used to process the audio files from the 3CX recordings bucket and convert them to text.
"""
import datetime
import json
import os

# Imports
import boto3
import openai
from dotenv import load_dotenv

# Global variables
bucket = None
bucket_path = None
bucket2 = None
data_path = None


def jsonl_append(filename, output, tokens, recoding_date, time, cost, size):
    """
    Appends the text to a JSONL single file as {time: text, filename: text, output: text}
    :param filename: The name of the file that is being processed.
    :param output: The text that is being appended to the JSONL file.
    :param tokens: The number of tokens that were used to generate the text.
    :param recoding_date: The date of the recording.
    :param time: The time it took to process the file.
    :param cost: The cost of processing the file.
    :param size: The size of the audio file in MB.
    :return: None
    """
    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime('%Y/%m/%d %H:%M:%S')
    recoding_date = recoding_date.strftime('%Y/%m/%d')
    log = {
        'transcription_date': timestamp,
        'filename': filename,
        'size': size,
        'time': time,
        'tokens': tokens,
        'cost': cost,
        'recording_date': recoding_date,
        'transcript': output,
        'summary': '',
        'question': '',
        'answer': '',
        'status': 'pending'
    }
    with open(data_path + 'data.jsonl', 'a') as f:
        json.dump(log, f)
        f.write('\n')


def config():
    """
    Loads the environment variables for OpenAI API key and AWS credentials.
    :return: None
    """
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_KEY')
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    s3 = boto3.resource(
        service_name='s3',
        region_name='eu-west-1',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    global bucket, bucket_path, bucket2, data_path
    bucket = s3.Bucket('recordings3cx')
    bucket_path = 'RU/7520/'
    bucket2 = s3.Bucket('3cxtranscriptions')
    data_path = '../data/'


def audio_to_text(filename):
    """
    Converts an audio file to text using OpenAI's Audio.transcribe method
    :param filename: The name of the file that is being processed.
    :return: The text that is being appended to the JSONL file.
    """
    with open(data_path + filename, "rb") as f:
        result = openai.Audio.translate("whisper-1", f)
    transcript = result["text"]
    os.remove(data_path + filename)
    return transcript


def main():
    """
    The main function that is called when the script is run.
    :return: None
    """
    config()
    start_file = datetime.datetime.now()
    for obj in bucket.objects.filter(Prefix=bucket_path):
        filename = obj.key.split('/')[-1]
        if os.path.exists(data_path + 'data.jsonl'):
            with open(data_path + 'data.jsonl', 'r') as f:
                if filename in f.read():
                    print(f"{datetime.datetime.now().replace(microsecond=0)} ({obj.size / 1e6:.2f} MB) already "
                          f"processed " + filename)
                    continue
        print(
            f"{datetime.datetime.now().replace(microsecond=0)} ({obj.size / 1e6:.2f} MB) working on " + filename)
        if (filename == "") or (0.1 > (obj.size / 1e6) or (obj.size / 1e6) > 25):
            continue
        else:
            mp3_content = obj.get()['Body'].read()
            with open(data_path + filename, 'wb') as f:
                f.write(mp3_content)
            obj_date = obj.last_modified
            obj_size = f"{obj.size / 1e6:.2f}"
            start = datetime.datetime.now()
            transcript = audio_to_text(filename)
            end = datetime.datetime.now()
            tokens = (end - start).seconds * 200
            cost = f"{(end - start).seconds * 0.006:.3f}"
            jsonl_append(filename, transcript, tokens, obj_date, (end - start).seconds, cost, obj_size)
            start_file = datetime.datetime.now()
    bucket2.upload_file(data_path + 'data.jsonl', '7520.jsonl')


if __name__ == "__main__":
    main()

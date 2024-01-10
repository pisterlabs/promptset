import openai
import json, re, time
import os
import random
import subprocess

import codecs
from boto3 import Session
from boto3 import resource
from boto3 import client

s3 = resource('s3')
bucket_name = 'gpt3-video-scripts'
bucket = s3.Bucket(bucket_name)
# Create an S3 client
s3_client = client('s3')
transcription_client = client('transcribe')

def search_items_in_bucket(bucket_name, folder_name):
    # List objects in the bucket
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=folder_name
        )

        # Check if objects were found
        if 'Contents' in response:
            # print(response['Contents'])
            matching_items = [obj['Key'] for obj in response['Contents'] if re.match(r'^\d+\.mp3$', obj['Key'].split('/')[-1])]
            # Sort items based on the numeric part of their names
            matching_items.sort(key=lambda x: int(re.search(r'\d+', x.split('/')[-1]).group()))
            return matching_items
        else:
            return []

    except Exception as e:
        print(f"Error listing objects: {e}")
        return []

def download_file_from_s3(bucket, key, local_path):
    s3_client.download_file(bucket, key, local_path)

def combine_audio_files(input_files, output_file, random_seed):
    # Check if there are input files
    if not input_files:
        print("No input files provided.")
        return
            
    # Create a temporary directory to store downloaded files
    tmp_dir = f'/tmp/audio_files{random_seed}'
    os.makedirs(tmp_dir, exist_ok=True)

    # Download input files from S3 to the temporary directory
    local_files = []
    for s3_key in input_files:
        local_path = os.path.join(tmp_dir, os.path.basename(s3_key))
        download_file_from_s3(bucket_name, s3_key, local_path)
        local_files.append(local_path)

    output_path = f"/tmp/output{random_seed}.mp3"
    # Build the ffmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-i', 'concat:' + '|'.join(local_files),
        '-c', 'copy',
        output_path
    ]

    try:
        # Run the ffmpeg command
        subprocess.run(ffmpeg_command, check=True, stdin=subprocess.DEVNULL)
        print(f"Audio files combined successfully. Output saved to {output_file}")
        # upload the output file to S3
        s3_client.upload_file(output_path, bucket_name, output_file)
    except subprocess.CalledProcessError as e:
        print(f"Error combining audio files: {e}")

def caption_audio_file(input_file, output_file, random_seed):
    job = transcription_client.start_transcription_job(
        TranscriptionJobName=f'transcription-job-{random_seed}',
        LanguageCode='en-US',
        MediaFormat='mp3',
        Media={
            'MediaFileUri': input_file
        },
        OutputBucketName=bucket_name,
        OutputKey=output_file,
        Subtitles={
          'Formats': [
              'srt',
          ],
        },
    )
    while True:
        status = transcription_client.get_transcription_job(TranscriptionJobName=f'transcription-job-{random_seed}')
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            return
        time.sleep(1)

def lambda_handler(event, context):
    folder_name = event['folder_name']
    caption_enabled = event['caption_enabled']
    print("Caption enabled: ", caption_enabled)
    # generate an 8 digit number
    random_number = random.randint(10000000, 99999999)
    all_audios = search_items_in_bucket(bucket_name, folder_name + "/audio")
    combine_audio_files(all_audios, f'{folder_name}/output.mp3', random_number)
    
    # do captions
    if caption_enabled:
      caption_audio_file(f's3://{bucket_name}/{folder_name}/output.mp3', f'{folder_name}/caption', random_number)
    
    return {
        'statusCode': 200,
        'body': json.dumps({"folder_name": folder_name})
    }

import boto3
from botocore.exceptions import NoCredentialsError
import chardet
import openai


s3_access_key = "AKIAZTHHIOR4JJ5HLTUB"
s3_secret_access_key = "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"
s3_bucket_name = "learnmateai"
openai.api_key = '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'

mocknumber = 1

s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_access_key)

def detect_encoding(data):
    result = chardet.detect(data)
    return result['encoding']

prompt = "generate two new question paper by analysing the question papers given below"


def read_text_files_from_s3(bucket_name, folder_path, last_file=None, batch_size=3):
    print("function called\n")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
    text_files = []
    encoding = None
    if "Contents" in response:
        file_count = 0
        start_processing = False
        for obj in response["Contents"]:
            if obj["Key"].lower().endswith('.txt'):
                if last_file is None or start_processing:
                    file_obj = s3.get_object(Bucket=bucket_name, Key=obj["Key"])
                    file_content = file_obj['Body'].read()
                    if encoding is None:
                        encoding = detect_encoding(file_content)
                    file_content = file_content.decode(encoding)
                    text_files.append(file_content)
                    file_count += 1
                    if file_count >= batch_size:
                        start_processing = False
                        break
                elif obj["Key"] == last_file:
                    start_processing = True

        # Process any files found
        if text_files:
            process_batch(text_files, last_file)

        # Determine the last file processed
        if file_count >= batch_size:
            last_file = obj["Key"]
        else:
            last_file = None

    return last_file

def process_batch(text_files, last_file):
    data = ""
    batch_number = 1
    if last_file is not None:
        print(f"Batch starting after file: {last_file}")
    print(f"Batch {batch_number}:\n")
    for i, text_file in enumerate(text_files):
        print(f"Text file {i+1}")
        #print(text_file)
        data = data + text_file
        print()
    print(f"{batch_number} data \n-------------------------------------------\n ")
    global prompt
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt + data
                }
            ]
        )
    global mocknumber
    important_topics = response.choices[0].message.content
    print(important_topics)
    s3.put_object(
            Body=important_topics.encode(),
            Bucket=s3_bucket_name,
            Key=f'Mock_QuestionPapers/Mock{mocknumber}.txt'
        )
    batch_number += 1
    mocknumber += 1
    

# Usage example
bucket_name = s3_bucket_name
folder_path = "pyqs_txt/"
batch_size = 2

# Initial call
last_file = read_text_files_from_s3(bucket_name, folder_path, batch_size=batch_size)







# Subsequent calls until all files are processed
while last_file is not None:
    last_file = read_text_files_from_s3(bucket_name, folder_path, last_file, batch_size=batch_size)

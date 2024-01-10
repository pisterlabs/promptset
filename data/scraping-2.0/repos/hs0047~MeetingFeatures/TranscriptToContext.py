import openai
import boto3

def generate_conversation(meeting_notes):
    dynamicPrompt = '''
please provide me the meeting notes for this diary text in the form of parsed JSON, with the title as meeting note and content as the resulting meeting note: '''
    # Generate conversation context using OpenAI's GPT-3.5 model
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=dynamicPrompt + meeting_notes,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
    )

    return response.choices[0].text.strip()

def read_text_file_from_s3(bucket, key, aws_access_key, aws_secret_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    return content

def write_text_file_to_s3(bucket, key, content, aws_access_key, aws_secret_key):
    print(content)
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    s3.put_object(Bucket=bucket, Key=key, Body=content)

def search_files_in_bucket(bucket, search_term, aws_access_key, aws_secret_key):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
    response = s3.list_objects_v2(Bucket=bucket)
    list_of_files = []

    # Iterate over the objects in the bucket
    for obj in response['Contents']:
        file_name=(obj['Key'])

        # Check if the file contains the search term
        if search_term in file_name:
            list_of_files.append(file_name)


    return list_of_files

def contextFunctionality(inputFile=""):
    outputFile = inputFile.replace("Diarization_","TranscriptToContext_")
    # Read the input text file from S3
    meeting_notes = read_text_file_from_s3(bucket, inputFile, aws_access_key, aws_secret_key)
    # Generate conversation context
    conversation_context = generate_conversation(meeting_notes)
    # Write the output text file to S3
    write_text_file_to_s3(bucket, outputFile, conversation_context, aws_access_key, aws_secret_key)

# Example usage
bucket = 'meeting-bot-processed-files'
openai.api_key = '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'
aws_access_key = 'AKIAVSYBEZQGCM6UUNMM'
aws_secret_key = 't2cVl1bM77ZHzQtHgVd7swXbbaePoegPRU9ROaGc'
search_term= 'Diarization_'

# List files in the S3 bucket
list_of_files = search_files_in_bucket(bucket, search_term, aws_access_key, aws_secret_key)

# Print the file names
for file_name in list_of_files:
    print(file_name)
    contextFunctionality(file_name)
    print(file_name+ " Processed Successfully")


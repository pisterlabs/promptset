from fastapi import APIRouter
import boto3
import openai
import time

s3_access_key = ""
s3_secret_access_key = ""
s3_bucket_name = "learnmateai"

s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_access_key)

# Set up OpenAI API credentials
openai.api_key = ''

def batch_text(input_text, delimiter="TOPIC:"):
    batches = input_text.split(delimiter)
    cleaned_batches = [batch.strip() for batch in batches if batch.strip()]
    return cleaned_batches

def upload_to_s3(bucket_name, folder_name, file_name, content):
    s3 = boto3.client('s3')
    key = folder_name + '/' + file_name
    s3.put_object(Body=content, Bucket=bucket_name, Key=key)

app = APIRouter()

@app.get("/process_files")
def process_files(user: str):
    user=user+"/"
    # Function to read and process a file
    def process_file(file_name):
        # Read file from S3
        response = s3.get_object(Bucket='learnmateai', Key=user+'notes_txt/' + file_name)
        file_content = response['Body'].read().decode('utf-8')

        # Split file content into batches (adjust batch size as needed)
        batch_size = 3000
        batches = [file_content[i:i+batch_size] for i in range(0, len(file_content), batch_size)]

        # Process batches
        for batch in batches:
            # Send batch to OpenAI API

            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"divide the text topic wise (it should look like TOPIC:notes) notes should very breif and be created in a way so that you will be able to recreate the full txt :\n\n{batch}\n\n"
                    }
                ]
            )

            important_topics = response.choices[0].message.content
            #print(important_topics)
            #return important_topics
            # Add a delay of 20 seconds to handle rate limit
            time.sleep(20)
            
            text_batches = batch_text(important_topics)

            bucket_name = 'learnmateai'
            file=file_name.split(".")[0]
            folder_name = f'{user}Analysed_Notes/{file}'

            for i, batch in enumerate(text_batches):
                lines = batch.split('\n')
                file_name1 = lines[0].strip().replace(" ", "_") + '.txt'
                content = '\n'.join(lines[1:]).strip()
                upload_to_s3(bucket_name, folder_name, file_name1, content)

                # Print uploaded file information
                print(f"File '{file_name1}' uploaded to '{bucket_name}/{folder_name}'")

    # Get the list of files in the "notes_txt" folder
    response = s3.list_objects_v2(Bucket='learnmateai', Prefix=user+'notes_txt/')

    # Process each file
    for file in response['Contents']:
        file_name = file['Key'].split('/')[-1]
        process_file(file_name)

    return {"message": "NOTES"}

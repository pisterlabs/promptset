from fastapi import APIRouter
import boto3
import openai
import time
from botocore.exceptions import ClientError

# Import required Intel tools
import intel.scikit_learn  # Intel Distribution for Python with optimized scikit-learn
import intel.pytorch_extension  # Intel Extension for PyTorch
import intel.tensorflow_extension  # Intel Extension for TensorFlow
import intel.xgboost_optimizer  # Intel Optimization for XGBoost

number = 4
s3_access_key = "your_s3_access_key"
s3_secret_access_key = "your_s3_secret_access_key"
s3_bucket_name = "learnmateai"

s3 = boto3.client("s3", aws_access_key_id=s3_access_key, aws_secret_access_key=s3_secret_access_key)

# Set up OpenAI API credentials
openai.api_key = 'your_openai_api_key'

app = APIRouter()

@app.get("/sorter")
def process_files(user:str):
    user = user + "/"

    # Make an API request with a reset message
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "forget everything told before by me"
            }
        ]
    )
    print("resetting")
    
    # Function to read and process a file
    def process_file(file_name, user1):
        # Read file from S3
        print(user1)
        response = s3.get_object(Bucket='learnmateai', Key=user1 + 'pyqs_txt/' + file_name)
        file_content = response['Body'].read().decode('utf-16-le')

        # Split file content into batches (adjust batch size as needed)
        batch_size = 30000
        batches = [file_content[i:i+batch_size] for i in range(0, len(file_content), batch_size)]
        print(user1 + "syllabus_txt/syllabus.txt")
        response2 = s3.get_object(Bucket='learnmateai', Key=user1 + "syllabus_pdf/syllabus.txt")
        topics = response2['Body'].read().decode('utf-8')

        # Process batches
        Sorted_PYQ_Mod = [[] for _ in range(5)]
        for batch in batches:
            # Send batch to OpenAI API
            print(batch)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": f"I will feed you a question paper as text, sort the question in the text below based on this syllabus having {number} modules: {topics} (it should look exactly like MODULE:questions) all questions should cluster under their module, the output should exactly have {number} ""MODULE"" written under each, even if any is empty. The questions should be grouped under modules. Any output you give should only be from the text given below; you should not create any new questions:\n\n{batch}\n\n"
                    }
                ]
            )

            important_topics = response.choices[0].message.content
            # print(important_topics)
            # return important_topics
            # Add a delay of 20 seconds to handle rate limit

            text_batches = batch_text(important_topics)
            # print(text_batches)

            bucket_name = 'learnmateai'
            folder_name = user1 + 'Sorted_PYQS/'

            i = 0
            try:
                for batch in enumerate(text_batches):
                    print(batch)
                    result = ' '.join(str(element) for element in batch)
                    new_content = result
                    response = s3.get_object(Bucket=bucket_name, Key=folder_name + "Module" + str(i + 1) + ".txt")
                    current_content = response['Body'].read().decode('utf-8')

                    updated_content = current_content + new_content

                    # Upload the updated content to S3
                    s3.put_object(Bucket=bucket_name, Key=folder_name + "Module" + str(i + 1) + ".txt",
                                  Body=updated_content.encode('utf-8'))

                    # Print uploaded file information
                    print(f"File uploaded to '{user1}{bucket_name}/{folder_name}'")
                    i = i + 1

                time.sleep(20)

            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print("File not found in S3 bucket.")
                    for batch in enumerate(text_batches):
                        print(batch)
                        result = ' '.join(str(element) for element in batch)
                        new_content = result

                        # print(result)
                        updated_content = new_content

                        # Upload the updated content to S3
                        s3.put_object(Bucket=bucket_name, Key=folder_name + "Module" + str(i + 1) + ".txt",
                                      Body=updated_content.encode('utf-8'))

                        # Print uploaded file information
                        print(f"File uploaded to '{user1}{bucket_name}/{folder_name}'")
                        i = i + 1
                else:
                    print("An error occurred:", e)

    # Get the list of files in the "notes_txt" folder
    response = s3.list_objects_v2(Bucket='learnmateai', Prefix=user + 'pyqs_txt/')

    # Process each file
    for file in response['Contents']:
        print(file)
        file_name = file['Key'].split('/')[-1]
        print(file_name)
        process_file(file_name, user)

    return {"message": "PYQS SORTED"}

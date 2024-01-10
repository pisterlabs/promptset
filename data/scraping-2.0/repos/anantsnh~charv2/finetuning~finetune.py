import openai
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# Prompt the user to enter the filename
filename = input("Enter the name of the file (without .json extension): ")
file_path = os.path.join("..", "data", "jsonl_files", filename + ".jsonl")

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File {file_path} does not exist. Exiting...")
    exit()

# Upload the data file for fine-tuning
response = openai.File.create(
    file=open(file_path, "rb"),
    purpose='fine-tune'
)

file_id = response['id']
print(f"File uploaded with ID: {file_id}")

# Check the status of the uploaded file
while True:
    file_status = openai.File.retrieve(file_id)['status']
    
    if file_status == "processed":
        print(f"File {file_id} has been processed and is ready for fine-tuning.")
        break
    elif file_status == "failed":
        print(f"File processing for {file_id} failed.")
        exit()
    else:
        print(f"File {file_id} is still being processed. Waiting...")
        time.sleep(30)

# edit file_id if file is already uploaded
# file_id = "file-sZ7y9pwtfcWzgVShE30QejVz"

# Start the fine-tuning job
response = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
job_id = response['id']
print(f"Fine-tuning job started with ID: {job_id}")

# Provide updates on the job status every 30 seconds
while True:
    response = openai.FineTuningJob.retrieve(job_id)
    print(f"Current response for job {job_id}: {response}")

    # Extract the status from the response
    status = response.get('status', None)
    
    if not status:
        print("Couldn't find the job status in the response. Exiting...")
        break

    print(f"Current status of the job {job_id}: {status}")
    
    # If job is completed or failed, exit the loop
    if status in ["completed", "failed"]:
        break
        
    time.sleep(30)

print("Fine-tuning process ended.")



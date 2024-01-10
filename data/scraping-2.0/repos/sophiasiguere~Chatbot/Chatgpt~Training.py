import openai
import time
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

if not api_key:
    raise Exception("API_KEY is not set")

# Configura tu clave de API
openai.api_key = api_key

# Sube el archivo JSONL
file_upload = openai.File.create(file=open("C:/Users/jseba/Documents/GitHub/Chatbot/Chatgpt/prueba.jsonl", "rb"), purpose="fine-tune")
print("Uploaded file id", file_upload.id)

while True:
    print("Waiting for file to process...")
    try:
        file_handle = openai.File.retrieve(id=file_upload.id)
        
        # Check if the file_handle is not empty and has a "processed" status
        if file_handle and file_handle.status == "processed":
            print("File processed")
            break
        elif file_handle and file_handle.status == "failed":
            print("File processing failed")
            break
    except Exception as e:
        # Handle any exceptions that may occur during the file retrieval
        print(f"Error while retrieving file status: {str(e)}")
        break

    # delay
    time.sleep(10)  

# Crea un trabajo de fine-tuning
job = openai.FineTuningJob.create(training_file=file_upload.id, model="gpt-3.5-turbo")

while True:
    print("Waiting for fine-tuning to complete...")
    job_handle = openai.FineTuningJob.retrieve(id=job.id)
    if job_handle.status == "succeeded":
        print("Fine-tuning complete")
        print("Fine-tuned model info", job_handle)
        print("Model id", job_handle.fine_tuned_model)

        with open("model.txt", "w") as model_file:
            model_file.write(job_handle.fine_tuned_model)
        break
    elif job_handle.status == "failed":
        print("Fine-tuning failed")
        break
    else:
        # Add a delay before checking the status again to avoid excessive API requests
        time.sleep(100)  # You can adjust the sleep time as needed

import requests
import time
import openai
import datetime
import json
import sys
# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')


def validate_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            the_file = [json.loads(line) for line in lines]
            return True
    except Exception as e:
        print("Error reading file, invalid format: ", e)
        return False


def upload_file():
    ft_file = openai.File.create(file=open("data.jsonl", "rb"), purpose='fine-tune')
    return ft_file["id"]


def create_model(training_file_id):
    ft_job = openai.FineTuningJob.create(training_file=training_file_id, model="gpt-3.5-turbo-0613")
    return ft_job["id"] 


def test_model(model_id, prompt):
    completion = openai.ChatCompletion.create(
        model=model_id,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful and professional customer service representative"},
            {"role": "user", "content": prompt},
        ]
    )

    print(completion.choices[0].message)


##############################
######### Main Logic #########
##############################

def main():
    # Validate the file
    is_file_valid = validate_file("data.jsonl")
    if is_file_valid == False:
        print("File is not valid")
        sys.exit()

    print("\nFile is valid and now uploading...\n")

    # Upload the file and wait for it to be processed
    file_id = upload_file()
    sleep_time_file = 5
    looptime = 0
    while True:
        print(f"Waiting for OpenAI to process the file... {looptime}")
        file_status = openai.File.retrieve(file_id)
        if file_status["status"] == "processed":
            print(f"\nFile processed: {file_status['id']}\n")
            break
        looptime += sleep_time_file
        time.sleep(sleep_time_file)

    # Create the finetuned model and wait for it to be processed
    fine_tuning_job = create_model(file_id)
    model_id = ""
    looptime = 0
    sleep_time_model = 30
    while True:
        print(f"Waiting for OpenAI to create the model... {looptime}")
        model_status = openai.FineTuningJob.retrieve(fine_tuning_job)
        if model_status["status"] == "succeeded":
            model_id = model_status["fine_tuned_model"]
            break
        looptime += sleep_time_model
        time.sleep(sleep_time_model)

    print(f"\nModel created: {model_id}")

    # Test the model
    print("\nTesting the new OpenAI model\n")
    prompt = "Where do I mail my check?"
    print(f"Prompt: {prompt}")
    test_model(model_id, prompt) 


if __name__ == "__main__":
    main()
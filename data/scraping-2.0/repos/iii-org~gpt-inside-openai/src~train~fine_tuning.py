import os
import time
import json
import shutil
import openai
import pandas as pd

from openai import cli


def check_status(training_id, validation_id):
    train_status = openai.File.retrieve(training_id)["status"]
    valid_status = openai.File.retrieve(validation_id)["status"]
    print(f'Status (training_file | validation_file): {train_status} | {valid_status}')
    return (train_status, valid_status)

def main(raw_data):
    training_file_name = '/tmp/training.jsonl'
    validation_file_name = '/tmp/validation.jsonl'

    ref_format_data = [{"prompt": f"根據以下參考資料回答，{d['Q']}\n參考資料：\n{d['REF']}", "completion": f"{d['A']}"} for d in raw_data.iloc]
    no_ref_format_data = [{"prompt": f"{d['Q']}", "completion": f"{d['A']}"} for d in raw_data.iloc]
    format_data = ref_format_data + no_ref_format_data

    # Generate the training dataset file.
    print(f'Generating the training file: {training_file_name}')
    with open(training_file_name, 'w') as training_file:
        for entry in format_data:
            json.dump(entry, training_file)
            training_file.write('\n')

    # Copy the validation dataset file from the training dataset file.
    # Typically, your training data and validation data should be mutually exclusive.
    # For the purposes of this example, you use the same data.
    print(f'Copying the training file to the validation file')
    shutil.copy(training_file_name, validation_file_name)

    # Upload the training and validation dataset files to Azure OpenAI.
    training_id = cli.FineTune._get_or_upload(training_file_name, True)
    validation_id = cli.FineTune._get_or_upload(validation_file_name, True)

    # Check the upload status of the training and validation dataset files.
    (train_status, valid_status) = check_status(training_id, validation_id)

    # Poll and display the upload status once per second until both files succeed or fail to upload.
    while train_status not in ["succeeded", "failed"] or valid_status not in ["succeeded", "failed"]:
        time.sleep(1)
        (train_status, valid_status) = check_status(training_id, validation_id)

    # This example defines a fine-tune job that creates a customized model based on curie,
    # with just a single pass through the training data. The job also provides
    # classification-specific metrics by using our validation data, at the end of that epoch.
    create_args = {
        "training_file": training_id,
        "validation_file": validation_id,
        "model": "gpt-35-turbo",
        "n_epochs": 1,
    }

    # Create the fine-tune job and retrieve the job ID and status from the response.
    resp = openai.FineTune.create(**create_args)
    job_id = resp["id"]
    status = resp["status"]

    # You can use the job ID to monitor the status of the fine-tune job.
    # The fine-tune job might take some time to start and complete.
    print(f'Fine-tuning model with job ID: {job_id}.')

    # Get the status of our fine-tune job.
    status = openai.FineTune.retrieve(id=job_id)["status"]

    # If the job isn't yet done, poll it every 2 seconds.
    if status not in ["succeeded", "failed"]:
        print(f'Job not in terminal status: {status}. Waiting.')
        while status not in ["succeeded", "failed"]:
            time.sleep(2)
            status = openai.FineTune.retrieve(id=job_id)["status"]
            print(f'Status: {status}')
    else:
        print(f'Fine-tune job {job_id} finished with status: {status}')

    # Check if there are other fine-tune jobs in the subscription.
    # Your fine-tune job might be queued, so this is helpful information to have
    # if your fine-tune job hasn't yet started.
    print('Checking other fine-tune jobs in the subscription.')
    result = openai.FineTune.list()
    print(f'Found {len(result)} fine-tune jobs.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='xlsx檔案儲存位置', required=True)
    args = parser.parse_args()
    data_path = args.data

    # Provide the key for your Azure OpenAI resource.
    # Remember to remove your key from your code when you're done.
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Provide the endpoint for your Azure OpenAI resource.
    # Example: https://<your-resource-name>.openai.azure.com/
    openai.api_base = os.getenv('OPENAI_API_BASE')
    openai.api_type = os.getenv('OPENAI_API_TYPE')

    # Provide the API version.
    # Note that the API version might change in the future.
    openai.api_version = os.getenv('OPENAI_API_VERSION')

    qa_df = pd.read_excel(
            data_path,
            converters={'Q': str.strip, 'A': str.strip, 'REF': str.strip}
            )
    main(qa_df)

import os
import openai
import time

openai.api_key = os.getenv("OPENAI_API_KEY")

def upload_training_data_to_openai(file_name):
    """
    Upload training data to open ai

    openai.File.create() returns:
    {
        "id": "file-abc123",
        "object": "file",
        "bytes": 120000,
        "created_at": 1677610602,
        "filename": "my_file.jsonl",
        "purpose": "fine-tune",
        "format": "fine-tune-chat",
        "status": "uploaded",
        "status_details": null
    }
    """
    res = openai.File.create(
        file=open(file_name, "rb"),
        purpose='fine-tune'
    )
    uploaded_file_id = res['id']

    while True:
        time.sleep(5)
        res = openai.File.retrieve(uploaded_file_id)
        if res['status'] == 'processed':
            break
        print('waiting for processing')
    return uploaded_file_id

def exec_fine_tuning(file_id):
    return openai.FineTuningJob.create(
        training_file=file_id,
        model="gpt-3.5-turbo",
        suffix="ISandT",
        hyperparameters={
            "n_epochs": 6
        }
    )

def main():
    file_name = "finetuning_train.jsonl"
    file_id = upload_training_data_to_openai(file_name)
    print('file_id is', file_id)
    exec_fine_tuning(file_id)

if __name__ == '__main__':
    main()

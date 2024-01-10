import os
import openai
import time


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    resp = openai.File.create(
        file=open("mydata.jsonl", "rb"),
        purpose='fine-tune'
    )
    training_file_id = resp['id']
    print(training_file_id)

    # Wait for the file to be processed
    time.sleep(60)

    resp = openai.FineTuningJob.create(training_file='file-ECeKUqgu1APeEXbDlWPi1tDH', model="gpt-3.5-turbo")
    job_id = resp['id']
    print(resp)

    # Retrieve the state of a fine-tune
    time.sleep(300)
    resp = openai.FineTuningJob.retrieve(job_id)
    while resp['status'] in ['running', 'created']:
        time.sleep(60)
        resp = openai.FineTuningJob.retrieve(job_id)
        print(resp)
        
    if resp['status'] != 'succeeded':
        raise Exception("Fine-tuning failed: " + resp['error'])
    
    model_name = resp['fine_tuned_model']
    print(model_name)
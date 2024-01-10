import openai


training_file_id = "file-pSC4mIivoxGuOsGLf1lBlomZ"
validation_file_id = "file-nhDhfhVKC0MNODkwB15wusn5"

create_args = {
    "training_file": training_file_id,
    "validation_file": validation_file_id,
    "model": "davinci",
    "n_epochs": 15,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
}

response = openai.FineTune.create(**create_args)
job_id = response["id"]
status = response["status"]

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")
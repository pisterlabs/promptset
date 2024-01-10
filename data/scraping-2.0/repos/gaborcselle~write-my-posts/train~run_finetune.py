import openai
import config
import common

openai.api_key = config.OPENAI_API_KEY

with open(common.TRAINING_SUPERSMALL_FILE_NAME, "rb") as training_fd:
    training_response = openai.files.create(
        file=training_fd, purpose="fine-tune"
    )

training_file_id = training_response.id

with open(common.VALIDATION_SUPERSMALL_FILE_NAME, "rb") as validation_fd:
    validation_response = openai.files.create(
        file=validation_fd, purpose="fine-tune"
    )
validation_file_id = validation_response.id

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)

response = openai.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix="wmp-no-replies",
)

job_id = response.id

print("Job ID:", response.id)
print("Status:", response.status)
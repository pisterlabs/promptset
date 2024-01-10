import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# Start the fine-tuning process
response = openai.FineTuningJob.create(
    training_file="file-REDJdFC5LaVirnVOizXHHyxQ",
    validation_file="file-4fi9X61Xr71vmJIvABxLEzQi",
    model="ft:gpt-3.5-turbo-0613:ultraware:pt-3:82cTtTbP",
    hyperparameters={"n_epochs": 1},
    suffix="pt-4"
)

print(response)

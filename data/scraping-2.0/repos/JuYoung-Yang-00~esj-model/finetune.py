import os
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.File.create(
    file=open("val_data.jsonl", "rb"),
    purpose='fine-tune'
    )
print(response)


ft_job_response = openai.FineTuningJob.create(
        training_file="file-JgWPlelnfOfMVbWeABEVDH8h",
        validation_file = "file-wrDwIMHdSBRiePQKj0HfPVj4", 
        model="gpt-3.5-turbo", 
        hyperparameters={
            "n_epochs":3,
            "batch_size":3,
        }
    )
print (ft_job_response)



# Retrieve the state of a fine-tune
state = openai.FineTuningJob.retrieve("ftjob-fkzRi46lpRdW5xaMFmUK6gl9")
print(state)
# Cancel a job
openai.FineTuningJob.cancel("ftjob-fkzRi46lpRdW5xaMFmUK6gl9")

# List up to 10 events from a fine-tuning job
events = openai.FineTuningJob.list_events(id="ftjob-fkzRi46lpRdW5xaMFmUK6gl9", limit=10)
print(events)
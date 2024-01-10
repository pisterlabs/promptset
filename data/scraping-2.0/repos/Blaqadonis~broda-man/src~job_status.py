import openai
import os
import getpass

# OpenAI API key
if os.getenv("OPENAI_API_KEY") is None:
    if any(['VSCODE' in x for x in os.environ.keys()]):
        print('Please enter password in the VS Code prompt at the top of your VS Code window!')
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI Key from: https://platform.openai.com/account/api-keys\n")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")

assert os.getenv("OPENAI_API_KEY", "").startswith("sk-"), "This doesn't look like a valid OpenAI API key"
print("OpenAI API key configured")


# Import the fine-tuning job ID from the environment
job_id = os.environ["BRODAMAN_FINETUNE_JOB_ID"]

# Check the status of the fine-tuning job
job = openai.FineTuningJob.retrieve(id=job_id)


# Print the status
print("Job Status:", job.status)
print("Model ID:", job.model)



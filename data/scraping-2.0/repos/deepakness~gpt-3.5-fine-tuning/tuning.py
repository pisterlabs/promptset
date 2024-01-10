import openai

openai.api_key = "YOUR_OPENAI_API_KEY"

def start_finetuning_job(file_id, model="gpt-3.5-turbo"):
    try:
        job = openai.FineTuningJob.create(training_file=file_id, model=model)
        print(f"Fine-tuning job created successfully: {job}")
        return job
    except Exception as e:
        print(f"Failed to create fine-tuning job. Error: {e}")
        return None

start_finetuning_job("file-AIvPJuN78Mtl1BWzkmtngFAj")
import openai_secret_manager

assert "openai" in openai_secret_manager.get_services()
secrets = openai_secret_manager.get_secret("openai")

openai.api_key = secrets["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"]

# Create a file with your training data
file = openai.File.create(
    file=open("training-data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create a fine-tuning job
job = openai.FineTuningJob.create(
    training_file=file["id"],
    model="gpt-3.5-turbo"
)

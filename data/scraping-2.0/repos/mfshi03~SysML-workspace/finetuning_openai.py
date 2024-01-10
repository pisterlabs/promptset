import openai
import os
from dotenv import load_dotenv

def create_fine_tune():
  client = openai.OpenAI()
  client.files.create(
    file=open("data/finetune.jsonl", "rb"),
    purpose="fine-tune"
  )

  client.fine_tuning.jobs.create(
    training_file="file-abc123", 
    model="gpt-3.5-turbo"
  )


# List 10 fine-tuning jobs
def fine_tuning_funcs():
  client.fine_tuning.jobs.list(limit=10)

  # Retrieve the state of a fine-tune
  client.fine_tuning.jobs.retrieve("ftjob-abc123")

  # Cancel a job
  client.fine_tuning.jobs.cancel("ftjob-abc123")

  # List up to 10 events from a fine-tuning job
  client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

  # Delete a fine-tuned model (must be an owner of the org the model was created in)
  client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")

response = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo-0613:credits::8YHIXDx8",
  messages=[
    {"role": "system", "content": "SystemGPT is a chatbot that evaluate master plans in the form: UGV(unmanned grounded vehicle) must traverse a [rough] terrain to complete its mission in [x] hours and [y] lifetime cycles."},
    {"role": "user", "content": "UGV must traverse a beach terrain to complete its mission in 5 hours and 100 lifetime cycles."}
  ]
)

print(response)
print(response.choices[0]["message"]["content"])
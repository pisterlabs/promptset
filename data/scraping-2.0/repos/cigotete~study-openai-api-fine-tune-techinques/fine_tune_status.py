import os
import pprint
from openai import OpenAI
client = OpenAI()

pp = pprint.PrettyPrinter(indent=2)

# List 10 fine-tuning jobs
job_list = client.fine_tuning.jobs.list(limit=10)
result_dict = vars(job_list)
pp.pprint(result_dict)
print('#'*50)

# Retrieve the state of a fine-tune
specific_job = client.fine_tuning.jobs.retrieve("ftjob-xyz")
result_specific_job = vars(specific_job)
pp.pprint(result_specific_job)
print('#'*50)

# Cancel a job
#client.fine_tuning.jobs.cancel("ftjob-xyz")

# List up to 10 events from a fine-tuning job
specific_job_events = client.fine_tuning.jobs.list_events("ftjob-xyz", limit=10)
result_specific_job_events = vars(specific_job_events)
pp.pprint(result_specific_job_events)
print('#'*50)

# Delete a fine-tuned model (must be an owner of the org the model was created in)
#client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:xyz")
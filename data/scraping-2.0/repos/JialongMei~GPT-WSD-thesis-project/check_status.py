import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
job = openai.FineTune.retrieve(id="fine-tuning job ID")
print(job)

#check fine-tuning progress from openai servers
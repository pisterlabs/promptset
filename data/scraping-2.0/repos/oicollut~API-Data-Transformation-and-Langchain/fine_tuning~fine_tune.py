import openai
import os
openai.api_key = os.getenv("OPENAI_API_KEY")

"""openai.File.create(
  file=open("50_test_data.jsonl", "rb"),
  purpose='fine-tune'
)"""""


#openai.FineTuningJob.create(training_file="file-CgX9EbA0caPSldpZrg6HjOzA", model="gpt-3.5-turbo")


#openai.api_key = os.getenv("OPENAI_API_KEY")
#print(openai.File.list())

print(openai.FineTuningJob.list())
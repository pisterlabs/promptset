import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# print(openai.File.create(
#     file=open("./fine-tuning-data.jsonl", "rb"),
#     purpose='fine-tune'
# ))

# print(openai.FineTuningJob.create(
#     training_file="file-uCU5Z0il2Xxj6mtjk2j8GORn", model="gpt-3.5-turbo"))

# print(openai.FineTuningJob.list(limit=10))

import os
import openai

#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"
r = openai.FineTuningJob.create(training_file="file-8dAokqYUdvkQ5p0E6Asm9I1R", model="gpt-3.5-turbo", suffix="test-baustore-prod")

print(r)
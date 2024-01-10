import openai
import os 

print(openai.api_key)
os.environ["OPENAI_API_KEY"] = "your_api_key"
openai.api_key = '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'
print("after")
print(openai.api_key)

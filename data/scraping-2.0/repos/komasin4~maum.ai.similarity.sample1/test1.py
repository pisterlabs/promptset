from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv(verbose=True)

print(os.getenv("AZURE_OPENAI_KEY"))
print(os.getenv("AZURE_OPENAI_API_VERSION"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))

client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_KEY"),  
    api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
)

response = client.chat.completions.create(
    #model="gpt-3.5-turbo",
    model="gpt-4",
    messages=[
        {"role": "system", "content": "질문에서 제품명을 추출합니다."},
        {"role": "user", "content": "제주보리차가 얼마인지 알려주세요."}
    ]
)

print(response)
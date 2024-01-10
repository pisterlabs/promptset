
from openai import AzureOpenAI
import os
import json
from dotenv import load_dotenv
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData




user_prompt = "What is John's job in the movie John Wick?"

def get_safety_classification(response):
    if response.hate_result:
        print(f"Hate severity: {response.hate_result.severity}")
    if response.self_harm_result:
        print(f"SelfHarm severity: {response.self_harm_result.severity}")
    if response.sexual_result:
        print(f"Sexual severity: {response.sexual_result.severity}")
    if response.violence_result:
        print(f"Violence severity: {response.violence_result.severity}")


client = AzureOpenAI(
  azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"], 
  api_key=os.environ["AZURE_OPENAI_KEY"],  
  api_version="2023-07-01"
)

response = client.chat.completions.create(
    model="get54TurboDply", # model = "deployment_name".
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'{user_prompt}'},

    ]
)

answer = response['choices'][0]['message']['content']
request = AnalyzeTextOptions(text=answer)
print(answer)


#########   validate content safety   #########

# Create a Content Safety client and authenticate with Azure Key Credential
client = ContentSafetyClient(
    #endpoint=os.getenv("AZURE_ACSAFETY_ENDPOINT"),
    endpoint="https://content-filter-serv.cognitiveservices.azure.com/",
    #credential=AzureKeyCredential(os.getenv("AZURE_ACSAFETY_KEY"))
    credential=AzureKeyCredential("a513c2e433714257b587bb8946207dd5")
)

# Analyze text
try:
    txt_response = client.analyze_text(request)
    get_safety_classification(txt_response)
except HttpResponseError as e:
    print("Analyze text failed.")
    if e.error:
        print(f"Error code: {e.error.code}")
        print(f"Error message: {e.error.message}")
        raise
    print(e)
    raise

#############################################
print("#############################################")

image_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".\img\image2.jpg"))

with open(image_path, "rb") as file:
    img_request = AnalyzeImageOptions(image=ImageData(content=file.read()))
    
try:
    img_response = client.analyze_image(img_request)
    get_safety_classification(img_response)
except HttpResponseError as e:
    print("Analyze text failed.")
    if e.error:
        print(f"Error code: {e.error.code}")
        print(f"Error message: {e.error.message}")
        raise
    print(e)
    raise
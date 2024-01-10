#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://rcgth-hackathon-aoai-eus.openai.azure.com/"
openai.api_version = "2023-06-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Image.create(
    prompt='water bottle',
    size='1024x1024',
    n=1
)

image_url = response["data"][0]["url"]

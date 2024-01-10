import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2023-07-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = 'poc-aoai-deployment-1'  # This will correspond to the custom name you chose for your deployment when you deployed a model.

# Send a completion call to generate an answer
print('Sending a test completion job')
start_phrase = 'Write a tagline for an ice cream shop. '
response = client.completions.create(model=deployment_name, prompt=start_phrase, max_tokens=800,
                                     temperature=0.7,
                                     top_p=0.95,
                                     frequency_penalty=0,
                                     presence_penalty=0,
                                     stop=None)
print(response.choices[0].text)


# Output format:
# Clinic Name:
# Services:
# Location Latitude: 3.139003
# Location Longitude: 101.807124
# Operation Time:
#
#
# Below is the output format, and nothing else:
# Clinic Name:
# Services:
# Location Latitude:
# Location Longitude:
# Operation Time:
#
# I'm looking for Dental Crowns
# My Location Latitude: 3.139003
# My Location Longitude: 101.529464
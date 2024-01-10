from azure.identity import DefaultAzureCredential
import openai
import os

def main():
  default_credential = DefaultAzureCredential()
  token = default_credential.get_token('https://cognitiveservices.azure.com/.default')


  openai.api_type = 'azure_ad'
  openai.api_base = os.environ['API_BASE_URL']
  openai.api_version = '2023-03-15-preview'
  #openai.api_version = '2022-12-01'
  openai.api_key = token.token

  response = openai.Completion.create(
    #engine=os.environ['MODEL_DEPLOYMENT_NAME'],
    deployment_id=os.environ['MODEL_DEPLOYMENT_NAME'],
    prompt=f'{os.environ["PROMPT_TEXT"]}\n\n',
    temperature=float(os.environ['TEMPERATURE']),
    max_tokens=int(os.environ['MAX_TOKENS']),
    top_p=float(os.environ['TOP_P']),
    frequency_penalty=float(os.environ['FREQUENCY_PENALTY']),
    presence_penalty=float(os.environ['PRESENCE_PENALTY']),
    stop=None)
  
  print(response.choices[0].text)


if __name__ == '__main__':
  main()
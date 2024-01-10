import os
from openai import AzureOpenAI
import prompt_generator_for_gpt 
import sys


def promptOnlyText(gender = 'man', ethnicity = 'european', age = '30s', destination = 'Paris', minTemp = 10, maxTemp = 25, minPrec = 0, maxPrec = 15, sunnyDays = 5):
    string_setup = f'A {gender}, {ethnicity}, in their {age} is visiting {destination}. The temperature will be between {minTemp} and {maxTemp} degrees celsius and the precipitation between {minPrec} and {maxPrec} and {sunnyDays} sunny days. '
    
    string_text = f'As text output, list the outer set of clothes with basic descriptions (type of clothing, 2-3 word description) that would be fitting for searching this clothings online. Make sure that the text output is **only** a csv compatible output and no other text.'
    
    string_prompt = string_setup + string_text
    
    return string_prompt

## more details on https://github.com/openai/openai-python
def getClothingSuggestions(prompt_str):
    AZURE_CH_ENDPOINT = 'https://switzerlandnorth.api.cognitive.microsoft.com/'
    fname_CHATGPT_KEY = 'CHATGPT_TOKEN.txt' # TODO: change to your own API key. This is located under Home > Azure AI Services | Azure OpenAI > hackathon-hack-openai-10 > Keys and Endpoint > Key 1
    if os.path.isfile(fname_CHATGPT_KEY):
        with open(fname_CHATGPT_KEY, 'r') as fh:
            AZURE_CHATGPT_API_KEY = fh.read()
    else:
        print('Error: AZURE_CHATGPT_API_KEY file not found')
    
    client = AzureOpenAI(
      azure_endpoint = AZURE_CH_ENDPOINT, #os.getenv("AZURE_OPENAI_ENDPOINT"), 
      api_key = AZURE_CHATGPT_API_KEY, # os.getenv("AZURE_OPENAI_KEY"),  
      api_version="2023-05-15"
    )
    
    response = client.chat.completions.create(
        model="gpt-35-turbo", # model = "deployment_name".
        # model='gpt-4', ## better, but a lot slower, and more expensive
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            # {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            # {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            # {"role": "user", "content": "Do other Azure AI services support this too?"}
            {"role": "user", "content": prompt_str}
        ]
    )
    
    response_txt = response.choices[0].message.content
    return response_txt


##########START HERE
##########START HERE
if len(sys.argv) > 1:
    criteria = sys.argv[1]
else:
    criteria = "gender = 'man', ethnicity = 'Swiss', age = '50s', destination = 'Edinborough', minTemp = -5, maxTemp = 5, minPrec = 0, maxPrec = 15, sunnyDays = 5"

prompt_str = promptOnlyText(criteria)
clothingSuggestions = getClothingSuggestions(prompt_str)
print("\n"*3)
print(clothingSuggestions)




import os
from openai import OpenAI
from PIL import Image
import base64
import json
import requests

client = OpenAI(api_key = "YOUR KEY HERE")          #openai key

asticaAPI_key = 'YOUR KEY HERE'  # visit https://astica.ai      #astica key
asticaAPI_timeout = 25 # in seconds. "gpt" or "gpt_detailed" require increased timeouts
asticaAPI_endpoint = 'https://vision.astica.ai/describe'
asticaAPI_modelVersion = '2.1_full' # '1.0_full', '2.0_full', or '2.1_full'

# vision parameters:  https://astica.ai/vision/documentation/#parameters
asticaAPI_visionParams = 'gpt,describe,objects,faces'  # comma separated, defaults to "all". 
asticaAPI_gpt_prompt = '' # only used if visionParams includes "gpt" or "gpt_detailed"
asticaAPI_prompt_length = '90' # number of words in GPT response

'''    
    '1.0_full' supported visionParams:
        describe
        objects
        categories
        moderate
        tags
        brands
        color
        faces
        celebrities
        landmarks
        gpt               (Slow)
        gpt_detailed      (Slower)

    '2.0_full' supported visionParams:
        describe
        describe_all
        objects
        tags
        describe_all 
        text_read 
        gpt             (Slow)
        gpt_detailed    (Slower)
        
    '2.1_full' supported visionParams:
        Supports all options 
        
'''

def asticaAPI(endpoint, payload, timeout):
    response = requests.post(endpoint, data=json.dumps(payload), timeout=timeout, headers={ 'Content-Type': 'application/json', })
    if response.status_code == 200:
        return response.json()
    else:
        return {'status': 'error', 'error': 'Failed to connect to the API.'}


""""
# print API output
print('\nastica API Output:')
print(json.dumps(asticaAPI_result, indent=4))
print('=================')
print('=================')
# Handle asticaAPI response
if 'status' in asticaAPI_result:
    # Output Error if exists
    if asticaAPI_result['status'] == 'error':
        print('Output:\n', asticaAPI_result['error'])
    # Output Success if exists
    if asticaAPI_result['status'] == 'success':
        if 'caption_GPTS' in asticaAPI_result and asticaAPI_result['caption_GPTS'] != '':
            print('=================')
            print('GPT Caption:', asticaAPI_result['caption_GPTS'])
        if 'caption' in asticaAPI_result and asticaAPI_result['caption']['text'] != '':
            print('=================')
            print('Caption:', asticaAPI_result['caption']['text'])
        if 'CaptionDetailed' in asticaAPI_result and asticaAPI_result['CaptionDetailed']['text'] != '':
            print('=================')
            print('CaptionDetailed:', asticaAPI_result['CaptionDetailed']['text'])
        if 'objects' in asticaAPI_result:
            print('=================')
            print('Objects:', asticaAPI_result['objects'])
else:
    print('Invalid response')
"""

def get_image_base64_encoding(image_path: str) -> str:
    """
    Function to return the base64 string representation of an image
    """
    with open(image_path, 'rb') as file:
        image_data = file.read()
    image_extension = os.path.splitext(image_path)[1]
    base64_encoded = base64.b64encode(image_data).decode('utf-8')
    return f"data:image/{image_extension[1:]};base64,{base64_encoded}"

def chatWithChat(query):
   response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": query}]
    )
   response_text = response.choices[0].message.content
   return response_text
   
# Originally main
def main():
    with open("output.txt") as f:
        data = f.read()
    asticaAPI_input = data
    #with open("output.txt") as f:
    #  data = f.read()
    #asticaAPI_input = data
    # Define payload dictionary
    asticaAPI_payload = {
        'tkn': asticaAPI_key,
        'modelVersion': asticaAPI_modelVersion,
        'visionParams': asticaAPI_visionParams,
        'input': asticaAPI_input,
    }
    
    # call API function and store result
    asticaAPI_result = asticaAPI(asticaAPI_endpoint, asticaAPI_payload, asticaAPI_timeout)

    query = "Give me random name(s) (first and last), age group(s) (child, teen, adult, or senior), gender(s) (man/woman), object name(s) (e.g. balloon, water bottle), personality attributes (e.g. charismatic, pessimistic), and appearance(s) for each of the following inanimate objects while also not changing them to humans: "
    for i in range(len(asticaAPI_result['objects'])):
        query = query + asticaAPI_result['objects'][i]['name'] + " "
    query = query + ". Use the following caption to help decide their personality. "
    query = query + asticaAPI_result['caption']['text']
    query = query + " Then, formulate a beginning prompt for a story speaking from the narrator's perspective using the objects as characters. "
    response_text = chatWithChat(query)       # query is what you are asking chatgpt
    print(response_text)
    return response_text

if __name__ == "__main__":
    main()

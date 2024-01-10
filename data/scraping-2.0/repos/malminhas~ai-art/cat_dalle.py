import os
import openai
import requests
import getpass, os

def configureEnvVars(key, org_id):
    # NB: host url is not prepended with \"https\" nor does it have a trailing slash.
    os.environ['OPENAI_API_KEY'] = key
    os.environ['OPENAI_ORG_ID'] = org_id

def initDalle():
    configureEnvVars(getAPIKey('.openAIKey'),getOrganisationId('.openAIOrg'))
    openai.organization = os.getenv('OPENAI_ORG_ID')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    models = openai.Model.list()

def getAPIKey(file):
    with open(file) as f:
        key = f.read()
    return key

def getOrganisationId(file):
    with open(file) as f:
        key = f.read()
    return key

def textToImage(prompt_text, target, engine='dalle', show=False):
    response = openai.Image.create(
      prompt=prompt_text,
      n=1,
      size="1024x1024"
    )
    image_url = response['data'][0]['url']
    img_data = requests.get(image_url).content
    with open(target, 'wb') as handler:
        handler.write(img_data)
    return image_url
    
    
if __name__ == '__main__':
    initDalle()
    #prompt_text = 'a road sign pointing to the right with an arrow showing the way ahead with desert in the background in a photorealistic style'
    #prompt_text = 'festive new year scene in a photorealistic style'
    #prompt_text = 'computerised yin and yang photorealistic style'
    #prompt_text = 'an image showing allocation in photorealistic style'
    #prompt_text = 'an image demonstrating the principle of single threaded ownership in photorealistic style'
    #prompt_text = 'an image demonstrating computer engineering productivity in photorealistic style'
    #prompt_text = 'an image showing engineering managers collaborating photorealistic style'
    #prompt_text = 'an image showing a structural transformation in photorealistic style'
    #prompt_text = 'a scenic landscape showing seasonal change in progress in photorealistic style'
    #prompt_text = 'an image showing a happy homeowner pillar'
    prompt_text = 'an image showing a happy female Technical Program Manager'
    image_file = 'test_dalle.png'
    image_url = textToImage(prompt_text, image_file)
    print(f'generated image {image_url} of size {os.path.getsize(image_file)}')
import argparse
import os
import json

import flickrapi
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import wikipedia
from wikipedia import PageError
import warnings

import openai


def create_parser():
    parser = argparse.ArgumentParser(
        description='Fetches the title, tags, and the URL of an 400 pixel wide '
                    'image version with the Flickr API, generates a description '
                    'for the image with Azure Vision and OpenAI GPT-3, and '
                    'optionally sets the title and description with the Flickr '
                    'API.')
    parser.add_argument('photo_id', help='The ID of the photo on Flickr.')
    parser.add_argument('--api-key', help='The Flickr API key.')
    parser.add_argument('--api-secret', help='The Flickr API secret.')
    parser.add_argument('--azure-key', help='The Azure Vision API key.')
    parser.add_argument('--azure-endpoint', help='The Azure Vision API endpoint.')
    parser.add_argument('--gpt3-engine', default='text-davinci-002',
                        help='The GPT-3 engine to use.')
    parser.add_argument('--openai-api-key', help='The OpenAI API key.')
    parser.add_argument('--store', action='store_true',
                        help='If specified, the configuration file is written '
                             'and the program ends.')
    return parser


def get_flickr_image_info(api_key, api_secret, photo_id):
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    info = flickr.photos.getInfo(photo_id=photo_id)
    title = info['photo']['title']['_content']
    description = info['photo']['description']['_content']
    flickr_tags = info['photo']['tags']['tag']
    sizes = flickr.photos.getSizes(photo_id=photo_id)
    for size in sizes['sizes']['size']:
        if size['width'] >= 400 and size['width'] < 600:
            image_url = size['source']
            break
    return title, description, flickr_tags, image_url


def generate_title(objects, tags, engine):
    prompt=f'Create a short title for an image with these tags {", ".join(tags)}'
    print(f"Title prompt: {prompt}")
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    description = response['choices'][0]['text']
    if description.startswith('rera\n\n'):
        description = description[6:]
    return description

def generate_description(objects, title, tags, engine):
    wikipedia_link, description = get_wikipedia_link(title)
    if not description:
        prompt=f'Create an encyclopedic photo description for a photo titled \"{title}\". The photo has these tags: {", ".join(tags)}.'
        print(f"Description prompt: {prompt}")
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=1024,
            temperature=0.8,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        description = response['choices'][0]['text']
        if description.startswith('rera\n\n'):
            description = description[6:]
    if wikipedia_link:
        description = description + "\n\n\nWikipedia: " + wikipedia_link
        
    return description

def generate_tags(description, engine):
    prompt=f'suggest up to 20 comma separated image tags based on the following description: {description}'
    print(f"Description prompt: {prompt}")
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.8,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    tags = response['choices'][0]['text']
    return tags.split(",")


def get_azure_image_tags(azure_key, azure_endpoint, image_url):
    azure_credential = CognitiveServicesCredentials(azure_key)
    client = ComputerVisionClient(azure_endpoint, azure_credential)
    image_analysis = client.analyze_image(image_url,visual_features=[VisualFeatureTypes.tags, VisualFeatureTypes.brands, VisualFeatureTypes.objects])
    return [tag for tag in image_analysis.tags if tag.confidence], [object.object_property for object in image_analysis.objects if object.confidence]

def get_wikipedia_link(text):
    warnings.catch_warnings()
    warnings.simplefilter("ignore")
    try:
        result = wikipedia.search(text)
    except wikipedia.exceptions.DisambiguationError as e:
       print(e.options)
       return "", ""
    if result:
        try:
            pageid = result[0]
            page = wikipedia.page(pageid)
            return page.url, wikipedia.summary(pageid)
        except PageError:
            return "", ""
        except wikipedia.exceptions.DisambiguationError as e:
            print(e.options)
            return "", ""
        else:
            return "", ""   
 

def merge_tags(tags1, tags2):
    merged_tags = []
    if tags2: 
        for tag in tags2:
            if tag.confidence > 0.9 and tag.name not in merged_tags \
                and tag.name not in ["cloud", "sky", "outdoor"]:
                merged_tags.append(tag.name)
    for tag in tags1:
        merged_tags.append(tag['_content'])
    return merged_tags


def set_flickr_image_info(api_key, api_secret, photo_id, title, description, merged_tags):
    flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
    if not flickr.token_valid(perms='write'):
        # Get a request token
        flickr.get_request_token(oauth_callback='oob')
        # Open a browser at the authentication URL. Do this however
        # you want, as long as the user visits that URL.
        authorize_url = flickr.auth_url(perms='write')
        print(authorize_url)
        # Get the verifier code from the user. Do this however you
        # want, as long as the user gives the application the code.
        verifier = str(input('Verifier code: '))
        # Trade the request token for an access token
        flickr.get_access_token(verifier)

    #oauth_token = flickr.token
    
    flickr.photos.setTags(photo_id=photo_id, tags=','.join(merged_tags))
    flickr.photos.setMeta(photo_id=photo_id, title=title, description=description)


def run(photo_id, api_key, api_secret, azure_key, azure_endpoint, openai_api_key, gpt3_engine):
    title, description, flickr_tags, image_url = get_flickr_image_info(api_key, api_secret, photo_id)
    if description and title:
        return
    openai.api_key = openai_api_key
    
    if  len(flickr_tags) == 0: 
      azure_tags, objects = get_azure_image_tags(azure_key, azure_endpoint, image_url)
    else: 
      azure_tags = objects = []
      
    merged_tags = merge_tags(flickr_tags, azure_tags)
    print(merged_tags)
    if not title or title.startswith('_MG') or title.startswith('IMG') or title.startswith('DSC') or title.endswith('_iOS'):
        new_title = generate_title(objects, merged_tags, gpt3_engine)
    else:
        new_title = title
    if not description:
       new_description = generate_description(objects, new_title, merged_tags, gpt3_engine)
    else:
      new_description = description
    new_tags = generate_tags(new_description, gpt3_engine)
    for tag in new_tags:
       if tag not in merged_tags:
          merged_tags.append(tag) 

    set_flickr_image_info(api_key, api_secret, photo_id, new_title, new_description, merged_tags)

def main():
    parser = create_parser()
    args = parser.parse_args()

    config_file = os.path.join(os.path.expanduser('~'), '.flickr_config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    openai_api_key = args.openai_api_key or config.get('openai_api_key')
    if not openai_api_key:
        openai_api_key = input('Enter your Open AI API key: ')
    api_key = args.api_key or config.get('api_key')
    if not api_key:
        api_key = input('Enter your Flickr API key: ')
    api_secret = args.api_secret or config.get('api_secret')
    if not api_secret:
        api_secret = input('Enter your Flickr API secret: ')
    azure_key = args.azure_key or config.get('azure_key')
    if not azure_key:
        azure_key = input('Enter your Azure Vision API key: ')
    azure_endpoint = args.azure_endpoint or config.get('azure_endpoint')
    if not azure_endpoint:
        azure_endpoint = input('Enter your Azure Vision API endpoint: ')

    config = dict({
        'api_key': api_key,
        'api_secret': api_secret,
        'azure_key': azure_key,
        'azure_endpoint': azure_endpoint,
        'openai_api_key': openai_api_key
        })
    if args.store:
        with open(config_file, 'w') as f:
            json.dump(config, f)
    else:
        run(args.photo_id, api_key, api_secret, azure_key, azure_endpoint, openai_api_key, args.gpt3_engine)

if __name__ == '__main__':
    main()
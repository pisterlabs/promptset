# Import necessary libraries
import openai
import sys
import json
import html
import re
import ssl
import os
import math
import glob
import pprint
import nltk
import pdb
import requests
import time
import random
from PIL import Image, ImageDraw, ImageFont
from PIL import UnidentifiedImageError
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt', quiet=True)
 
# Get the first command line argument
location = sys.argv[1]
sku = sys.argv[2]

# Initialize an empty dictionary for credentials
credentials = {}

# Define the file path to the credentials file
creds_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # Get the directory of the current file
    "../creds2.txt"  # Append the relative path to the credentials file
)
if os.path.exists('product.json'):
    os.remove('product.json')

# Define a class to represent a location
class Location:
    def __init__(self, website, user, city, phone, consumer_key, consumer_secret, api_key):
        self.website = website
        self.user = user
        self.city = city
        self.phone = phone
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.api_key = api_key  # Here's the new attribute

locations = []

# Open the credentials file
with open(creds_file_path) as f:
    # Initialize variables for parsing the file
    website = None
    user = None
    city = None
    phone = None
    consumer_key = None
    consumer_secret = None
    openai.api_key = None
    for line in f:
        line = line.strip()  # Remove trailing and leading whitespace
        if line.startswith("[") and line.endswith("]"):
            if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
                locations.append(Location(website, user, city, phone, consumer_key, consumer_secret, openai.api_key))
            website = line[1:-1].lstrip()  # Remove the brackets and any leading whitespace
            user = None
            city = None
            phone = None
            consumer_key = None
            consumer_secret = None
            openai.api_key = None
        elif line.startswith("["):
            website = line[1:]
        elif line.endswith("]"):
            website += line[:-1]
        elif website and " = " in line:
            key, value = line.split(" = ")
            if key == "user":
                user = value
            elif key == "city":
                city = value
            elif key == "phone":
                phone = value
            elif key.lower().endswith("_consumer_key"):
                consumer_key = value
            elif key.lower().endswith("_consumer_secret"):
                consumer_secret = value
            elif key == "openai.api_key":
                openai.api_key = value

    
    print(locations)    
    print(location)    
    pdb.set_trace()
    print(locations)    
    print(location)    
    if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
        locations.append(Location(website, user, city, phone, consumer_key, consumer_secret, openai.api_key))
    print(locations)    
    print(location)    
    pdb.set_trace()

        
def generate(new_pics_prompt):
    res = openai.Image.create(
        prompt=new_pics_prompt,
        n=1,
        size="1024x1024",
    )
    return res["data"][0]["url"]

def remove_keys(images_data):
    keys_to_remove = ['date_created', 'date_created_gmt', 'date_modified', 'date_modified_gmt', 'id', 'alt']
    new_images_data = []
    for index, image_data in enumerate(images_data):
        if index < 4:
            new_image_data = {key: value for key, value in image_data.items() if key not in keys_to_remove}
        else:
            new_image_data = {}
        new_images_data.append(new_image_data)
    return new_images_data


def add_watermark(image_url, watermark_text):
    try:
        # Download the image from the URL
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Open the downloaded image using PIL
        image = Image.open(response.raw)

        # Create a drawing object for the image
        draw = ImageDraw.Draw(image)

        # Define the font and size for the watermark
        font = ImageFont.truetype('path_to_font.ttf', size=40)

        # Calculate the width and height of the watermark text
        text_width, text_height = draw.textsize(watermark_text, font=font)

        # Calculate the position to place the watermark text (centered on the image)
        image_width, image_height = image.size
        x = (image_width - text_width) // 2
        y = (image_height - text_height) // 2

        # Apply the watermark by drawing the text on the image
        draw.text((x, y), text=watermark_text, font=font, fill=(255, 255, 255, 128))

        # Save the modified image (you can overwrite the original file or save to a new file)
        image.save('path_to_save_image.jpg')
        
    except Exception as e:
        print(f"Error adding watermark to image: {str(e)}")


#fetches the first product dataset to be edited and pushed to the other sites.
pdb.set_trace()
for location in locations:
    pdb.set_trace()
    base_url = "https://" + location.website + "/wp-json/wc/v3/products"
    consumer_key = location.website + "_consumer_key:" + location.consumer_key
    consumer_secret = location.website + "_consumer_secret:" + location.consumer_secret

    auth = (
        location.consumer_key,
        location.consumer_secret,
    )

    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()

    product = response.json()[0]
    source_product = product
    source_product['images'] = remove_keys(source_product['images'])
    
    pprint.pprint(product['images'])
    time.sleep(1)
    break

#fetches all but the first product and applies the updated first site product details.
pdb.set_trace()
for location in locations[1:]:
    pdb.set_trace()
    base_url = "https://" + location.website + "/wp-json/wc/v3/products"
    consumer_key = location.website + "_consumer_key:" + location.consumer_key
    consumer_secret = location.website + "_consumer_secret:" + location.consumer_secret
    auth = (
     location.consumer_key,
     location.consumer_secret,
           )
    opkey = openai.api_key
    city = location.city
    phone = location.phone

    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()
    product = response.json()[0]
    print("Add stuff to mix up name here.")
    product['name'] = source_product['name']
    print("Add stuff to mix up name here.")
    
    image_count = 0
# Update the product images with the new image
    for image in product['images']:
        pdb.set_trace()
        image_count = image_count + 1
        del image['id']
        del image['date_created']
        del image['date_created_gmt']
        del image['date_modified']
        del image['date_modified_gmt']
    print("Pre-existing image count", image_count)
    del product['images']
    product['images'] = source_product['images']

    pdb.set_trace()
    # Generate three new image URLs
    new_image_url1 = generate("Picture of a happy guy with a vape")
    new_image_url2 = generate("Picture of a happy girl smoking a joint")
    new_image_url3 = generate("Picture of a super stoned happy face!")
   
    # Add the new image URLs to the product['images'] array
    product['images'].append({'src': new_image_url1, 'name': 'Super happy stoner guy loves his vape!'})
    product['images'].append({'src': new_image_url2, 'name': 'Supewr happy stoner girl loves her joint!'})
    product['images'].append({'src': new_image_url3, 'name': 'Super Doap Stoney Cartoon Happy Face!'})

    new_short_description = source_product['short_description'] + " Get 1hr delivery bo calling  " + city.strip('"') + " Doap at " + phone.strip('"') + " anytime between 9-9 daily 7 days a week. We deliver to " + city.strip('"') + " and surrounding cities!" 
    new_short_description = new_short_description.replace("Alamo", city.strip('"'))
    product['short_description'] = new_short_description.replace("925-553-4710", phone.strip('"'))

    pdb.set_trace()
    # product['name'] = source_product['name']
    # new_short_description = source_product['short_description'].replace('phone', phone)
    # new_short_description = new_short_description.replace('city', city)
    # product['short_description'] = new_short_description
    # product['short_description'] = source_product['short_description']
    product['description'] = source_product['description']
    product['date_created'] = source_product['date_created']
    product['date_created_gmt'] = source_product['date_created_gmt']
    product['date_modified_gmt'] = source_product['date_modified_gmt']
    product['date_modified'] = source_product['date_modified']
    city = location.city
    phone = location.phone
    print("Processing: ",city)
    time.sleep(1)
    print("Setting source product title",product['name'], " on ", location.city)
    # print("source images",source_product['images'])
    # print()
    # print("current images",product['images'])
    print("City",city)
    #time.sleep(3)
    #pprint.pprint(product)
    #pprint.pprint(product)
    #time.sleep(3)
    pdb.set_trace()
    update_url = f'{base_url}/{product["id"]}'
    update_response = requests.put(update_url, json=product, auth=auth)
    update_response.raise_for_status()
    # pprint.pprint(product)
    pdb.set_trace()
    #break
    # time.sleep(30)

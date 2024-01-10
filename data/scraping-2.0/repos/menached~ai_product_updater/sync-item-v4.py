# Import necessary libraries
import openai
import sys
import json
import html
import re
import ssl
import os
import pprint
import nltk
import requests
import time
import random

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

    if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
        locations.append(Location(website, user, city, phone, consumer_key, consumer_secret, openai.api_key))
def generate(new_pics_prompt):
    res = openai.Image.create(
        prompt=new_pics_prompt,
        n=1,
        size="256x256",
    )
    return res["data"][0]["url"]



def remove_keys(images_data):
    keys_to_remove = ['date_created', 'date_created_gmt', 'date_modified', 'date_modified_gmt', 'id', 'alt']
    new_images_data = []
    for image_data in images_data:
        new_image_data = {key: value for key, value in image_data.items() if key not in keys_to_remove}
        new_images_data.append(new_image_data)
    return new_images_data



for location in locations:
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
    
    # Generate two new image URLs
    new_image_url1 = generate("Picture of a cat")
    new_image_url2 = generate("Picture of a dog")
    
    # Add the new image URLs to the product['images'] array
    product['images'].append({'src': new_image_url1, 'name': 'new-image-1'})
    product['images'].append({'src': new_image_url2, 'name': 'new-image-2'})

    pprint.pprint(product['images'])
    time.sleep(1)
    break

for location in locations[1:]:
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

    product['name'] = source_product['name']


    image_count = 0
# Update the product images with the new image
    for image in product['images']:
        image_count = image_count + 1
        del image['id']
        del image['date_created']
        del image['date_created_gmt']
        del image['date_modified']
        del image['date_modified_gmt']
    print("Image count", image_count)

    new_pic_prompts = [
                        f"Create a picture of this cannabis product: '{product['name']}'.", 
                        f"Create a picture of a very pretty girl delivering a tiny package to a handsome guy."
                       ]
    new_image_urls = [generate(prompt) for prompt in new_pic_prompts]

    for i, image in enumerate(product['images']):
        if i < len(new_image_urls):
            old_image_url = image['src']
            image['src'] = new_image_urls[i]
            print(f"Old Image URL: {old_image_url}")
            print(f"New Image URL: {image['src']}\n")
        else:
            break


    new_short_description = source_product['short_description'] + " Get 1hr delivery bo calling  " + city.strip('"') + " Doap at " + phone.strip('"') + " anytime between 9-9 daily 7 days a week. We deliver to " + city.strip('"') + " and surrounding cities!" 
    new_short_description = new_short_description.replace("Alamo", city.strip('"'))
    product['short_description'] = new_short_description.replace("925-553-4710", phone.strip('"'))

    # product['name'] = source_product['name']
    # new_short_description = source_product['short_description'].replace('phone', phone)
    # new_short_description = new_short_description.replace('city', city)
    # product['short_description'] = new_short_description
    # product['short_description'] = source_product['short_description']
    product['description'] = source_product['description']
    del product['images']
    del product['button_text']
    del product['sold_individually']
    del product['stock_quantity']
    del product['tax_class']
    del product['tax_status']
    del product['total_sales']
    del product['weight']
    del product['meta_data']
    del product['rating_count']
    del product['purchase_note']
    del product['shipping_class_id']
    del product['shipping_required']
    del product['shipping_taxable']
    product['date_created'] = source_product['date_created']
    product['date_created_gmt'] = source_product['date_created_gmt']
    product['date_modified_gmt'] = source_product['date_modified_gmt']
    product['date_modified'] = source_product['date_modified']
    del product['date_on_sale_from']
    del product['date_on_sale_from_gmt']
    del product['date_on_sale_to']
    del product['date_on_sale_to_gmt']
    del product['shipping_class']
    del product['default_attributes']
    del product['dimensions']
    del product['download_expiry']
    del product['download_limit']
    del product['downloadable']
    del product['downloads']
    del product['external_url']
    del product['grouped_products']
    del product['has_options']
    product['images'] = source_product['images']
    city = location.city
    phone = location.phone
    print("Processing: ",city)
    time.sleep(1)
    print("Setting source product title",product['name'], " on ", location.city)
    print("source images",source_product['images'])
    print()
    print("current images",product['images'])
    #print("phone",phone)
    #time.sleep(3)
    #pprint.pprint(product)
    #pprint.pprint(product)
    #time.sleep(3)
    update_url = f'{base_url}/{product["id"]}'
    update_response = requests.put(update_url, json=product, auth=auth)
    update_response.raise_for_status()
    # pprint.pprint(product)
    break
    # time.sleep(30)



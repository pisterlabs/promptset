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
            elif key == "website":
                website = value

    
    #if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
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
for location in locations:
    base_url = "https://" + location.website + "/wp-json/wc/v3/products"
    consumer_key = location.website + "_consumer_key:" + location.consumer_key
    consumer_secret = location.website + "_consumer_secret:" + location.consumer_secret
    city = location.city
    phone = location.phone
    website = location.website

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
    website = location.website

    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()
    product = response.json()[0]
    product['name'] = "Super duper " + source_product['name'] 
    
    pdb.set_trace()
    product['images'] = source_product['images']
    pdb.set_trace()
    #del product['images']
    del product['date_created']
    del product['date_created_gmt']
    del product['date_modified']
    del product['date_modified_gmt']

    image_count = 0
    for image in product['images']:
        image_count = image_count + 1
    
    # Generate new image URLs
    # Add them new image URLs to the product['images'] array
    pdb.set_trace()
    new_image_url1 = generate("Picture of a happy guy with a vape")
    pdb.set_trace()
    product['images'].append({'src': new_image_url1, 'name': 'Happy guy with a vape.'})
    pdb.set_trace()
    # new_image_url2 = generate("Picture of a happy girl smoking a joint")
    # product['images'].append({'src': new_image_url2, 'name': 'Happy girl smoking a joint.'})
    # new_image_url3 = generate("Picture of a super stoned happy face!")
    # product['images'].append({'src': new_image_url3, 'name': 'Super stoned happy face.'})

    new_short_description = source_product['short_description'] + " Get 1hr delivery bo calling  " + city.strip('"') + " Doap at " + phone.strip('"') + " anytime between 9-9 daily 7 days a week. We deliver to " + city.strip('"') + " and surrounding cities!" 
    new_short_description = new_short_description.replace("Alamo", city.strip('"'))
    product['short_description'] = new_short_description.replace("925-553-4710", phone.strip('"'))

    new_description = source_product['description'] + " Get 1hr delivery bo calling  " + city.strip('"') + " Doap at " + phone.strip('"') + " anytime between 9-9 daily 7 days a week. We deliver to " + city.strip('"') + " and surrounding cities!" 
    new_description = new_description.replace("Alamo", city.strip('"'))
    #product['description'] = new_description
    product['description'] = new_description.replace("925-553-4710", phone.strip('"'))
    product['date_created'] = source_product['date_created']
    product['date_created_gmt'] = source_product['date_created_gmt']
    product['date_modified_gmt'] = source_product['date_modified_gmt']
    product['date_modified'] = source_product['date_modified']
    city = location.city
    phone = location.phone
    print("Processing: ",city)
    print("Setting source product title",product['name'], " on ", location.city)
    #print("Images: ",product['images'])
    print("City: ",city)
    #pprint.pprint(product)
    del product['date_created']
    del product['date_created_gmt']
    del product['date_modified']
    del product['date_modified_gmt']
    del product['tax_class']
    del product['tax_status']
    del product['total_sales']
    del product['weight']
    del product['virtual']
    del product['variations']
    del product['upsell_ids']
    del product['stock_quantity']
    del product['stock_status']
    del product['sold_individually']
    del product['rating_count']
    del product['shipping_required']
    del product['shipping_taxable']
    del product['type']
    del product['reviews_allowed']
    del product['related_ids']
    del product['price_html']
    del product['parent_id']
    del product['shipping_class']
    del product['shipping_class_id']
    del product['status']
    del product['purchase_note']
    del product['purchasable']
    del product['meta_data']
    del product['low_stock_amount']
    del product['manage_stock']
    del product['menu_order']
    del product['dimensions']
    del product['download_expiry']
    del product['download_limit']
    del product['downloadable']
    del product['downloads']
    del product['external_url']
    del product['has_options']
    del product['grouped_products']
    del product['attributes']
    del product['average_rating']
    del product['backordered']
    del product['backorders']
    del product['backorders_allowed']
    del product['button_text']
    del product['catalog_visibility']
    del product['cross_sell_ids']
    del product['date_on_sale_from']
    del product['date_on_sale_from_gmt']
    del product['date_on_sale_to']
    del product['date_on_sale_to_gmt']
    del product['default_attributes']
    update_url = f'{base_url}/{product["id"]}'
    update_response = requests.put(update_url, json=product, auth=auth)
    update_response.raise_for_status()
    images = product['images']
    images = images[:3]
    product['images'] = images
    pdb.set_trace()
    pprint.pprint(product)
    #break
    # time.sleep(30)

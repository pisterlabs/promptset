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

class Location:
    def __init__(self, website, user, city, phone, consumer_key, consumer_secret, api_key):
        self.website = website
        self.user = user
        self.city = city
        self.phone = phone
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.api_key = api_key  # Here's the new attribute


def generate_new_product_name(sku):
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry.",
            },
            {
                "role": "user",
                "content": f"Use this product slug '{product['slug']}' to rewrite the product title. The slug contains words separated by a -."
            f"Use them to come up with a new name that is max 70 chars long and will rank well with regard to SEO. If there is a mention of price. Change it to some other descriptive language."
            },
        ]
    )

    new_product_name = ai_response['choices'][0]['message']['content'].strip()
    new_product_name = html.unescape(re.sub('<.*?>', '', new_product_name))

    return new_product_name


def generate_new_image_name(image_name):
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a creative AI assistant and California Budtender for a delivery service.",
            },
            {
                "role": "user",
                "content": f"I have an image with the name '{image_name}'. Please suggest a new name for the image that does not use dates or times in the name. Limit the name to 70 characters."
            },
        ]
    )

    new_image_name = ai_response['choices'][0]['message']['content'].strip()
    new_image_name = html.unescape(re.sub('<.*?>', '', new_image_name))

    return new_image_name



def remove_keys(images_data):
    keys_to_remove = [
    'date_created',
    'date_created_gmt',
    'date_modified',
    'date_modified_gmt',
    'id',
    'alt'
    ]   
    new_images_data = []
    for index, image_data in enumerate(images_data):
        if index < 4:
            new_image_data = {key: value for key, value in image_data.items() if key not in keys_to_remove}
        else:
            new_image_data = {}
        new_images_data.append(new_image_data)
    return new_images_data

def generate(new_pics_prompt):
    res = openai.Image.create(
        prompt=new_pics_prompt,
        n=1,
        size="256x256",
    )
    return res["data"][0]["url"]


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
    # pdb.set_trace()
    for line in f:
        # pdb.set_trace()
        line = line.strip()  # Remove trailing and leading whitespace
        # pdb.set_trace()
        if line.startswith("[") and line.endswith("]"):
            # pdb.set_trace()
            if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
                # pdb.set_trace()
                locations.append(Location(website, user, city, phone, consumer_key, consumer_secret, openai.api_key))
                # pdb.set_trace()
            website = line[1:-1].lstrip()  # Remove the brackets and any leading whitespace
            # pdb.set_trace()
            user = None
            city = None
            phone = None
            consumer_key = None
            consumer_secret = None
            openai.api_key = None
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
                aikey = value
            elif key == "website":
                website = value
     
    locations.append(
        Location(website, user, city, phone, consumer_key, 
                 consumer_secret, openai.api_key)
    )
        
#fetches the first product dataset to be edited and pushed to the other sites.
for locationa in locations:
        print(itemurl)


for locationb in locations[2:]:
    seq = seq + 1
    base_url = "https://" + locationb.website + "/wp-json/wc/v3/products"
    consumer_key = locationb.website + "_consumer_key:" + locationb.consumer_key
    consumer_secret = locationb.website + "_consumer_secret:" + locationb.consumer_secret
    city = locationb.city
    city = city.replace('"', '')
    phone = locationb.phone
    phone = phone.replace(' ', '').replace('-', '').replace('"', '').replace('(', '').replace(')', '')

    website = locationb.website
    aikey = openai.api_key

    auth = (
        locationb.consumer_key,
        locationb.consumer_secret,
    )

    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()

    product = response.json()[0]
    #source_product = product
    source_product['images'] = remove_keys(source_product['images'])
    print()
    msgg = "#" + str(seq) + " " + str(sku)
    print(msgg)
    subdomain = website.split('.')[0]
    print("Domain: ", subdomain)
    print(city, "Doap")
    print(city, " Ca ", phone)
    print("Sku: ", sku)
    print()
    print("Dest product name")
   
   # First AI call: generate new product name
    ##### product['name'] = generate_new_product_name(sku).replace('"','')
    print(product['name'])

    print()
    print("Images")
    print()
    imgcnt = 0
    for item in source_images:
        imgcnt = imgcnt + 1
        itemname = item['name'].replace('-',' ').capitalize()
        print("Image #", imgcnt)
        if  "Screen" in itemname:
            
            ##### new_unique_product_name = generate_new_image_name(itemname).replace('"','')
            new_unique_product_name = new_unique_product_name.replace('"','')

            print(new_unique_product_name)
            print(item['src'])
            image_url = item['src']
            # add_watermark(image_url, "Doap.com")
        else:
            
            ##### new_unique_product_name = generate_new_image_name(itemname).replace('"','')
            new_unique_product_name = itemname

            item['name'] = new_unique_product_name 
            print(new_unique_product_name)
            print(item['src'])

    #pprint.pprint(source_images)
    break

api_url = "https://campbell.doap.com/wp-json/wp/v2/media"
print()
print(api_url)
consumer_key_bits = consumer_key.split(':')
consumer_key = consumer_key_bits[1]
consumer_secret_bits = consumer_secret.split(':')
consumer_secret = consumer_secret_bits[1]

print(consumer_key)
print(consumer_secret) 
print(sku)

pdb.set_trace()
# Watermark details
watermark_text = "Doap"
font_size = 36  # Adjust as needed
font_color = (255, 255, 255)  # White color, you can change it as needed

# Retrieve all media items having the given SKU
params = {
    'consumer_key': consumer_key,
    'consumer_secret': consumer_secret,
    'per_page': 100,  # Increase if needed to retrieve all media items
    'search': sku,
}

pdb.set_trace()

response = requests.get(api_url, params=params)
try:
    media_items = response.json()
except requests.exceptions.JSONDecodeError:
    print(response.content)  # Print the response content for debugging purposes
    raise

# media_items = response.json()

# Iterate through each media item and add watermark
for media_item in media_items:
    image_url = media_item['media_details']['sizes']['full']['source_url']
    pdb.set_trace()

    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    pdb.set_trace()

    # Add the watermark text
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("font.ttf", font_size)  # Use a different font if needed
    text_width, text_height = draw.textsize(watermark_text, font)
    x = (image.width - text_width) // 2
    y = (image.height - text_height) // 2
    draw.text((x, y), watermark_text, font=font, fill=font_color)
    pdb.set_trace()

    # Save the watermarked image
    watermarked_image_path = f"watermarked_{media_item['id']}.jpg"
    image.save(watermarked_image_path)
    pdb.set_trace()

    # Upload the watermarked image back to the media item
    headers = {
        'Content-Disposition': f'attachment; filename={watermarked_image_path}',
        'Content-Type': 'image/jpeg',
    }
    with open(watermarked_image_path, 'rb') as file:
        response = requests.post(
            f"{api_url}/{media_item['id']}/attachments",
            params={'consumer_key': consumer_key, 'consumer_secret': consumer_secret},
            headers=headers,
            data=file.read(),
        )
    pdb.set_trace()

    # Remove the temporary watermarked image file
    os.remove(watermarked_image_path)
    pdb.set_trace()




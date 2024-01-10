# Import necessary libraries
import openai
import subprocess
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

sitelist = [
  { "subdomain": "alamo", "site_id": 29 },
  { "subdomain": "burlingame", "site_id": 30 },
  { "subdomain": "campbell", "site_id": 7 },
  { "subdomain": "castrovalley", "site_id": 25 },
  { "subdomain": "concord", "site_id": 31 },
  { "subdomain": "danville", "site_id": 9 },
  { "subdomain": "dublin", "site_id": 8 },
  { "subdomain": "hillsborough", "site_id": 12 },
  { "subdomain": "lafayette", "site_id": 13 },
  { "subdomain": "livermore", "site_id": 14 },
  { "subdomain": "orinda", "site_id": 34 },
  { "subdomain": "pittsburg", "site_id": 28 },
  { "subdomain": "pleasanthill", "site_id": 35 },
  { "subdomain": "sanramon", "site_id": 33 },
  { "subdomain": "walnutcreek", "site_id": 32 }
]


def get_site_id(subdomain):
  for site in sitelist:
    if site["subdomain"] == subdomain:
      return site["site_id"]
  return None

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

def scp_file_to_remote(local_file, remote_file):
    try:
        # Run SCP command
        subprocess.Popen(["scp", local_file, remote_file])
        print("File transfer initiated.")
        
    except Exception as e:
        print("Error while copying the file:", e)

def download_image(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Image downloaded successfully: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {str(e)}")



def add_watermark_and_save(image_path, watermark_text, output_path):
    try:    
        # Open the image
        image = Image.open(image_path).convert("RGBA")

        # Define the watermark text and font style
        font = ImageFont.truetype("font.ttf", 40)

        # Create a transparent overlay and draw the watermark text
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text_width, text_height = draw.textbbox((0, 0), watermark_text, font=font)[:2]
        # position = ((image.width - text_width) // 2, (image.height - text_height) // 2)
        position = (image.width - text_width - 10, image.height - text_height - 10) # Position the watermark in the lower right corner

        draw.text(position, watermark_text, font=font, fill=(128, 128, 128, 128))

        # Composite the image and watermark overlay
        watermarked = Image.alpha_composite(image, overlay)

        # Save the watermarked image with the specified output path
        watermarked.save(output_path)
        print(f"Watermarked image saved as {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


def makeunique(new_unique_product_name):
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry.",
            },
            {
                "role": "user",
                "content": f"Use this product name '{new_unique_product_name}'. Use this phrase to come up with a slightly different name that means the same thing."
            f"Come up with a new name that is max 70 chars long and will rank well with regard to SEO. If there is a mention of price. Change it to some other descriptive language instead."
            },
        ]
    )


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
            f"Use them to come up with a new name that is max 70 chars long and will rank well with regard to SEO. If there is a mention of price. Change it to some other descriptive language. Dont put spaces in the names. Use underscores to separate words."
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
                "content": f"I have an image with the name '{image_name}'. Please suggest a new name for the image that does not use dates or times in the name. Limit the name to 70 characters. Dont put spaces in the names. Use underscores to separate words."
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
for locationa in locations[:1]:
    base_url = "https://" + locationa.website + "/wp-json/wc/v3/products"
    consumer_key = locationa.website + "_consumer_key:" + locationa.consumer_key
    consumer_secret = locationa.website + "_consumer_secret:" + locationa.consumer_secret
    city = locationa.city
    phone = locationa.phone
    website = locationa.website
    aikey = openai.api_key

    auth = (
        locationa.consumer_key,
        locationa.consumer_secret,
    )

    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()

    product = response.json()[0]
    source_product = product
    source_product['images'] = remove_keys(source_product['images'])
    source_images = source_product['images'][:4]  
    imagecounter = 0
    for item in source_images:
        imagecounter = imagecounter + 1
        print("Image:",imagecounter)
        #source_product_name = product['name'].strip()
        item['src'] = item['src'].replace("/29/","/30/")
        item['src'] = item['src'].replace("alamo","burlingame")
        #imgcnt = 0
        #pprint.pprint(source_images)
        #source_image_url = item['src']
        # for item in source_images:
        # source_product_name = product['name'].strip()
        # print("Source Product\n",source_product_name)
        # print(website, aikey)
        # print("Source Images")
        # imgcnt = 0
        # pprint.pprint(source_images)
        # source_image_url = item['src']

# new_product_name = generate_new_product_name(sku)
# print("New name suggestion:", new_product_name)

seq = 0
#fetches all but the first product and applies the updated first site product details.
print("Destination Products\n")
for locationb in locations[1:]:
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
    product['images'] = source_product['images'] 
    msgg = "#" + str(seq) + " " + str(sku)
    print(msgg)
    subdomain = website.split('.')[0]
    print("Domain: ", subdomain)
    site_id = get_site_id(subdomain)
    print("Site ID:", site_id) 
    print(city, "Doap")
    print(city, " Ca ", phone)
    print("Sku: ", sku)
   # First AI call: generate new product name
    product['name'] = generate_new_product_name(sku).replace('"','').replace('"','').replace("'","").replace(" ","_").replace("(","").replace(")","").replace(",","").replace("$","")
    print("New dest product name: ", product['name'])
    print("New Images")
    imgcnt = 0
    for item in source_images:
        imgcnt = imgcnt + 1
        itemname = item['name'].replace('-',' ').capitalize()
        print("Image #", imgcnt)
        itemname = item['name'].replace('-',' ').capitalize()
        # print("Image #", imgcnt)
        new_unique_product_name = generate_new_image_name(product['name']).replace('"','').replace('"','').replace("'","").replace("!","").replace("(","").replace(")","").replace(",","").replace("→","")
        new_unique_file_name = new_unique_product_name
        item['name'] = new_unique_product_name
        # print(item['name'], " : ", item['src'])
        source_image_url = item['src']
        source_image_filename = os.path.basename(source_image_url)
        new_unique_file_name = new_unique_file_name + ".png"
        download_image(source_image_url, source_image_filename)
        print("Source image url: ", source_image_url)
        replaced_url = source_image_url.replace("https://alamo.", "/var/www/")
        stripped_path = "/".join(replaced_url.split("/")[:-1])
        print("Orig file path: ", stripped_path)
        
        new_path = stripped_path.split("/")
        new_path[7] = str(site_id)
        new_path = "/".join(new_path)

        print("New remote file path: ", new_path)
        #item['src'] = "https://" + subdomain + ".doap.com/" + stripped_path + "/" + new_unique_file_name
        item['src'] = "https://" + subdomain + ".doap.com/" + stripped_path + "/" + new_unique_file_name
        item['src'] = item['src'].replace("/var/www/doap.com/","")
        watermark_text = city + " Doap " + phone
        add_watermark_and_save(source_image_filename, watermark_text, new_unique_file_name)
        local_file = '/Users/dmenache/Nextcloud/Projects/doap-api/ai_product_updater/' + new_unique_file_name 
        remote_server = 'dmenache@debian.doap.com'
        testpath = stripped_path.replace("https://burlingame.","/var/www/")
        remote_file = f'{remote_server}:{testpath}/{new_unique_file_name}'
        scp_file_to_remote(local_file, remote_file)
        pdb.set_trace()
        #pprint.pprint(item)
        #pprint.pprint(source_images)
        product['images'] = source_images
        #pprint.pprint(product)
        # pprint.pprint(product)
        for image in product['images']:
            image['src'] = image['src'].replace('https://burlingame.doap.com/https://burlingame.doap.com/', 'https://burlingame.doap.com/')
        print("product[images]",product['images'])
        print("source_images",source_images)
        print("product[images]",product['images'])
    break
pprint.pprint(product)
pdb.set_trace()
update_url = f'{base_url}/{product["id"]}'
update_response = requests.put(update_url, json=product, auth=auth)
update_response.raise_for_status()



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

# Initialize an empty list to store locations
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
    # Loop over each line in the file
    for line in f:
        line = line.strip()  # Remove trailing and leading whitespace
        # If the line is a website (indicated by brackets), store it and reset other variables
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
        # If the line starts with a bracket but doesn't end with one, it's a multiline website; just store the first part
        elif line.startswith("["):
            website = line[1:]
        # If the line ends with a bracket but doesn't start with one, it's the end of a multiline website; append this part
        elif line.endswith("]"):
            website += line[:-1]
        # If the line contains " = ", it's a key-value pair; parse and store it
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

    # Once we've parsed the entire file, check if there are any leftover variables and, if so, add another location
    if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
        locations.append(Location(website, user, city, phone, consumer_key, consumer_secret, openai.api_key))

# Print the locations
for location in locations:
    print("Using " + location.website + " as source product." + sku)
    base_url = "https://" + location.website + "/wp-json/wc/v3/products"
    city = location.city
    phone = location.phone
    consumer_key = location.website + "_consumer_key:" + location.consumer_key
    consumer_secret = location.website + "_consumer_secret:" + location.consumer_secret

    auth = (
     location.consumer_key,
     location.consumer_secret,
           )
    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()

    if not response.json():
        print(f"No product found with SKU: {sku}")
        exit()

    product = response.json()[0]

    #time.sleep(1)
    # pprint.pprint(product)
    print("Source site: ", location.website)
    print("Product to clone: ", product['sku'], "Name: ", product['name']) 
    print()
    break    

time.sleep(1)
print("Turn AI loose...")
for location in locations[1:]:
    print("Using " + location.website + " as source product." + sku)
    base_url = "https://" + location.website + "/wp-json/wc/v3/products"
    consumer_key = location.website + "_consumer_key:" + location.consumer_key
    consumer_secret = location.website + "_consumer_secret:" + location.consumer_secret

    auth = (
     location.consumer_key,
     location.consumer_secret,
           )

    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()

    if not response.json():
        print(f"No product found with SKU: {sku}")
        exit()

    product = response.json()[0]

    sku = product['sku']
    city = location.city
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry in Northern California. ",
                },
            {
                "role": "user",
                "content": f"I have a product with SKU '{sku}' named '{product['name']}' with a short description of '{product['short_description']}'."
                f"I need a new but similar name for this product that will both help with SEO and improve the product visibility in search engines."
                f"Use the city name '{city}' in the title. Examples of really good usage would be variations of things like Best in Burlingame or San Ramons best. "
                f"Don't stray too far from the core idea of the original name.  Add the city name to the product title somehow. "
                f"Limit the new product name to about 70 characters.  Do not use any punctuation or apostrophes or any single or double quotes. "
                f"Use proper capitalization. Optimize all for SEO.  Never use prices in the new titles."
                },
            ]
        )

    new_product_name = response['choices'][0]['message']['content'].strip()
    new_product_name = html.unescape(re.sub('<.*?>', '', new_product_name))
    print("Suggested new product name: ", new_product_name)
    product['name'] = new_product_name

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry in and around the city of '{city}'. ",
                },
            {
                "role": "user",
                "content": f"I am a budtender looking to deliver cannabis products to my customers. "
                f"Create a list of 5 possible public meetup spots in the area of '{city}' that can be used as meetup spots if the customer doesnt want to meet at home or work. Prioritize discreet and easy access."
                f"The locations should be convenient, well known, and not near a school or police station. Present them as a bullet list with short descriptions as to why they are good meetup spots."
                },
            ]
        )
    
    meetup_spots = response['choices'][0]['message']['content'].strip()
    meetup_spots = html.unescape(re.sub('<.*?>', '', meetup_spots))
    product['description'] = product['description'] + " \n\n  Have your " + new_product_name +  " delivered to your home or work or choose one of these great meetup spots in " + city + "\n" +  meetup_spots
    #print("Suggested meetups spots for ",city,": ", meetup_spots)
    #print("New product description", product['description'])
    update_url = f'{base_url}/{product["id"]}'
    update_response = requests.put(update_url, json=product, auth=auth)
    update_response.raise_for_status()
    break


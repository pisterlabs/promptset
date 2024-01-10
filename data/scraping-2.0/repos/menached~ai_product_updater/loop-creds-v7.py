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
sku = sys.argv[1]

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
    print()
    # print("Config API and fetch product sku", sku, " from ", "https://"+website)
    # base_url = "https://" + website
    base_url = "https://" + location.website + "/wp-json/wc/v3/products"
    # print(base_url)
    # print(openai.api_key)
    # print(sku, location.website)
    city = location.city
    # print("City:", location.city)
    phone = location.phone
    # print("Phone:", location.phone)
    # print(location.website + "_consumer_key:", location.consumer_key)
    consumer_key = location.website + "_consumer_key:" + location.consumer_key
    # print(location.website + "_consumer_secret:", location.consumer_secret)
    consumer_secret = location.website + "_consumer_secret:" + location.consumer_secret
    # print(consumer_key) 
    # print(consumer_secret) 

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

    with open('product.json', 'w') as json_file:
        json.dump(product, json_file)

    time.sleep(3)
    pprint.pprint(product)

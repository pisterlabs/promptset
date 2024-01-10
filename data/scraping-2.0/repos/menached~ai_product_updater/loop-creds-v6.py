import openai
import sys
import json
import os
import pprint
import nltk
import requests

if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt', quiet=True)

sku = sys.argv[1]
credentials = {}

creds_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../creds2.txt"
)

class Location:
    def __init__(self, website, user, city, phone, consumer_key, consumer_secret):
        self.website = website
        self.user = user
        self.city = city
        self.phone = phone
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret

locations = []

with open(creds_file_path) as f:
    website = None
    user = None
    city = None
    phone = None
    consumer_key = None
    consumer_secret = None
    for line in f:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            if website and user and city and phone and consumer_key and consumer_secret:
                locations.append(Location(website, user, city, phone, consumer_key, consumer_secret))
            website = line[1:-1].lstrip()  # Remove leading space
            user = None
            city = None
            phone = None
            consumer_key = None
            consumer_secret = None
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

if website and user and city and phone and consumer_key and consumer_secret:
    locations.append(Location(website, user, city, phone, consumer_key, consumer_secret))

for location in locations:
    print()
    print(sku, location.website)
    print("City:", location.city)
    print("Phone:", location.phone)
    print(website, "_consumer_key:", location.consumer_key)
    print(website, "_consumer_key:", location.consumer_secret)

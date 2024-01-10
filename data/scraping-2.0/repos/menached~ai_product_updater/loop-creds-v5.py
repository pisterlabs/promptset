import openai
import sys
import json
import os
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
    def __init__(self, website, user, city, phone):
        self.website = website
        self.user = user
        self.city = city
        self.phone = phone

locations = []

with open(creds_file_path) as f:
    website = None
    user = None
    city = None
    phone = None
    for line in f:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            if website and user and city and phone:
                locations.append(Location(website, user, city, phone))
            website = line[1:-1]
            user = None
            city = None
            phone = None
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

if website and user and city and phone:
    locations.append(Location(website, user, city, phone))

for location in locations:
    print("Location:", location.website)
    print("User:", location.user)
    print("City:", location.city)
    print("Phone:", location.phone)
    print()


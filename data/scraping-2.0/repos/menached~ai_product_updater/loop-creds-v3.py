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
current_location = None  # Move this line outside the `with` block

with open(creds_file_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            if current_location:
                locations.append(current_location)
            website = current_location + ".doap.com"
            current_location = Location(website, "", "", "")
        elif line.startswith("["):
            current_location = line[1:]
        elif line.endswith("]"):
            current_location += line[:-1]
            locations.append(current_location)
        elif current_location and " = " in line:
            key, value = line.split(" = ")
            if key == "user":
                current_location.user = value
            elif key == "city":
                current_location.city = value
            elif key == "phone":
                current_location.phone = value

# Append the last location to the locations list
if current_location:
    locations.append(current_location)

# Print the locations array
for location in locations:
    print("Location:", location.website)
    print("User:", location.user)
    print("City:", location.city)
    print("Phone:", location.phone)
    print()

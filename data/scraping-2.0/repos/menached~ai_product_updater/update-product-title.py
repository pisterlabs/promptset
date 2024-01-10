import openai
import sys
import html
import re
import json
import os
import ssl
import nltk
import requests
from PIL import Image
import pprint

if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt', quiet=True)

location = sys.argv[1] + ".doap.com"

sku = sys.argv[2]

credentials = {}
creds_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../creds2.txt"
)

with open(creds_file_path) as f:
    current_section = None
    for line in f:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
        elif current_section == location:
            key, value = line.split(" = ")
            credentials[key] = value

openai.api_key = credentials["openai.api_key"]
city = credentials['city']
phone = credentials['phone']

auth = (
    credentials[location + "_consumer_key"],
    credentials[location + "_consumer_secret"]
)

def generate(new_pics_prompt):
    res = openai.Image.create(
        prompt=new_pics_prompt,
        n=1,
        size="256x256",
    )
    return res["data"][0]["url"]

base_url = "https://" + location + "/wp-json/wc/v3/products"

response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
response.raise_for_status()

if not response.json():
    print(f"No product found with SKU: {sku}")
    exit()

product = response.json()[0]

with open('product.json', 'w') as json_file:
    json.dump(product, json_file)

response = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    model="gpt-3.5-turbo",
    messages = [
    {
        "role": "system",
        "content": "You are a helpful budtender who knows all about the cannabis industry.",
    },
    {
        "role": "user",
        "content": f"Use this product slug '{product['slug']}' to rewrite the product title.  The slug contains words separated by a -."
                   f"Use them to come up with a new name that is max 70 chars long and will rank well with regard to SEO."
    },
]
)

new_product_name = response['choices'][0]['message']['content'].strip()
new_product_name = html.unescape(re.sub('<.*?>', '', new_product_name))

old_product_name = product['name']
product['name'] = new_product_name

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
                    f"Create a picture of a '{product['slug']}' keeping in mind this description '{product['short_description']}'.", 
                    f"Create a picture of this cannabis product: '{product['name']}'.", 
                    f"Create a picture of a very pretty girl delivering a tiny package to a handsome guy."
                    f"Create a picture of a hippie dancing."
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

print(
    f"SKU: {sku}\n\n"
    f"Old name:\n{old_product_name}\n"
    f"New name:\n{new_product_name}\n"
)

pprint.pprint (product)
# Update the product with the new name
update_url = f'{base_url}/{product["id"]}'
update_response = requests.put(update_url, json=product, auth=auth)
update_response.raise_for_status()

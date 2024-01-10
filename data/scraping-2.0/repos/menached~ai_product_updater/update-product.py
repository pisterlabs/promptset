import openai
import sys
import html
import re
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
        size="1024x1024",
    )
    return res["data"][0]["url"]

base_url = "https://" + location + "/wp-json/wc/v3/products"

response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
response.raise_for_status()

if not response.json():
    print(f"No product found with SKU: {sku}")
    exit()

product = response.json()[0]

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
        "content": f"I have a product with SKU '{sku}' named '{product['name']}' with a short description of '{product['short_description']}' and a long description of '{product['description']}'. "
                   f"I need a new but similar name for this product that will both help with SEO and improve the product visibility in search engines. "
                   f"Don't stray too far from the core idea of the original name.  Use the word Doap as an acronym for awesome. "
                   f"Limit the new product name to about 70 characters.  Do not use any punctuation or apostrophes or single or double quotes. "
                   f"Use proper capitalization. Optimize all for SEO.  Never use prices in the new titles."
                   f"Never spell the word dope, always substitute doap. Dont use the phone number in the title."
    },
]
)

new_product_name = response['choices'][0]['message']['content'].strip()
new_product_name = html.unescape(re.sub('<.*?>', '', new_product_name))

response_short_description = openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
    model="gpt-3.5-turbo",
    messages = [
    {
        "role": "system",
        "content": "You are a knowledgeable and creative budtender with a way with words. You are skilled in creating captivating descriptions for cannabis products. You are a bit of a hippie. You speak fairly plainly and tend to not make the descriptions too foo foo.",
    },
    {
        "role": "user",
        "content": f"I have a product with SKU '{sku}' named '{new_product_name}' with a short description '{product['short_description']}' and description '{product['description']}'. "
                   f"I need a new but similar short description that will both help with SEO and improve the product visibility in search engines."
                   f"Limit the new short description to about 160 characters."
                   f"Optimize for SEO and make sure it is not detectable by AI detectors. "
                   f"Doap is a synonym for awesome."
                   f"Mention for delivery of that awesome product anywhere in or around '{city}' call '{phone}' between 9am and 9pm 7 days a week."
                   f"Make sure it is not detectable by AI detectors. "
                   f"If its greenhouse flower mention the locally sourced organic flower from the city of '{city}' and if not flower mention locally produced."
                   f"Don't mention price."
    },
]
)

new_short_description = response_short_description['choices'][0]['message']['content'].strip()
new_short_description = html.unescape(re.sub('<.*?>', '', new_short_description))

response_long_description = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [
    {
        "role": "system",
        "content": "You are a knowledgeable and creative budtender with a way with words. You are skilled in creating engaging and detailed descriptions for cannabis products.",
    },
    {
        "role": "user",
        "content": f"I have a product with SKU '{sku}' named '{new_product_name}' with a short_description of '{new_short_description}' long description ' {product['description']}'. "
                   f"I need a new but similar long description that will both help with SEO and improve the product visibility in search engines. "
                   f"The new description should be detailed and captivating to pique the interest of potential customers. "
                   f"Optimize for SEO and make sure it is not detectable by AI detectors."
                   f"Mention we can meet up at certain landmarks in the '{city}' area. Mention call '{phone}' to get ultra fast 1hr delivery. "
                   f"and don't mention price anywhere. "
                   f"Come up with and casually mention a few landmarks that are convenient for quickly meeting up that are well-known, well-traveled in the '{city}' area. "
                   f"Casually mention we could meet you at one of these locations or deliver directly to your home or work. "
                   f"Mention for delivery of that awesome product anywhere in or around '{city}' call '{phone}' between 9am and 9pm 7 days a week."
                   f"Make sure it is not detectable by AI detectors. "
                   f"If its greenhouse flower mention the locally sourced organic flower from the city of '{city}' and if not flower mention locally sourced."
                   f"Mention we accept cash, credit, crypto, applepay, zelle, paypal, venmo and others."
                   f"Limit the new description to about 900 characters."
    },
]
)

new_long_description = response_long_description['choices'][0]['message']['content'].strip()
new_long_description = html.unescape(re.sub('<.*?>', '', new_long_description))

old_product_name = product['name']
product['name'] = new_product_name
old_short_description = product['short_description']
product['short_description'] = new_short_description

old_long_description = product['description']
product['description'] = new_long_description

for image in product['images']:
    del image['id']
    del image['date_created']
    del image['date_created_gmt']
    del image['date_modified']
    del image['date_modified_gmt']

new_pic_prompts = ["Create a picture of a happy guy holding a bag of weed.", 
                   "Create a picture of a cannabis bud up close.", 
                   "Create a picture of a cannabis plant up close.", 
                   "Create a picture of a cartoon bong on a black background.", 
                   "Create a picture of a very pretty girl delivering a tiny package to a handsome guy."]
new_image_urls = [generate(prompt) for prompt in new_pic_prompts]

for i, image in enumerate(product['images']):
    if i < len(new_image_urls):
        old_image_url = image['src']
        image['src'] = new_image_urls[i]
        # print(f"Old Image URL: {old_image_url}")
        # print(f"New Image URL: {image['src']}\n")
    else:
        break

print(
    f"SKU: {sku}\n\n"
    f"Old name:\n{old_product_name}\n"
    # f"Old short_description:\n{old_short_description}\n"
    # f"Old description:\n{old_long_description}\n"
    # f"New name:\n{new_product_name}\n"
    # f"New short_description:\n{new_short_description}\n"
    # f"New description:\n{new_long_description}"
)

# pprint.pprint (product)

# Update the product with the new name
update_url = f'{base_url}/{product["id"]}'
update_response = requests.put(update_url, json=product, auth=auth)
update_response.raise_for_status()

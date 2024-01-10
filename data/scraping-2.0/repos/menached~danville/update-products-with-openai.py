import openai
import os
import re
import ssl
import nltk
import requests

if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt', quiet=True)

credentials = {}
creds_file_path = os.path.join(
          os.path.dirname(os.path.abspath(__file__)),
          "../danville.txt"
          )

with open(creds_file_path) as f:
    current_section = None
    for line in f:
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
        elif current_section == "sanramon.doap.com":
            key, value = line.split(" = ")
            credentials[key] = value

base_url = "https://sanramon.doap.com/wp-json/wc/v3/products"

auth = (
    credentials["sanramon.doap.com_consumer_key"],
    credentials["sanramon.doap.com_consumer_secret"]
    )

page = 1
while True:
    response = requests.get(f'{base_url}?page={page}', auth=auth)
    response.raise_for_status()
    products = response.json()

    if not products:
        break

    for product in products:
        # Use OpenAI API to generate new product name
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful budtender who knows all about the cannabis industry."},
                {"role": "user", "content": f"I have a product named '{product['name']}' with a short description of '{product['short_description']}' and a long description of '{product['description']}'. I need a new but similar name for this product that will both help with SEO and improve the product visibility in search engines.  Dont stray too far from the core idea of the original description. Use the word Doap as an acronmy for awesome."},
            ]
        )
        
        new_product_name = response['choices'][0]['message']['content'].strip()
        
        product['name'] = new_product_name
        print(f'Product ID: {product["id"]}, '
              f'SKU: {product["sku"]}, '
              f'New Name: {product["name"]}, '   # Modified product name will be displayed
              f'Permalink: {product["permalink"]}')
              
        # Updating product name on the server
        update_url = f'{base_url}/{product["id"]}'
        update_response = requests.put(update_url, json=product, auth=auth)
        update_response.raise_for_status()
        
    page += 1
    break

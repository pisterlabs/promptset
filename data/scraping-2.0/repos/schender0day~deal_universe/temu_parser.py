import csv
import glob
import http.client
import json
import logging
import os
import time
import urllib
from urllib.parse import urlparse
import imgkit
import openai
import string
import pandas as pd
import pyperclip
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate
from test import extract_promotion_code_in_html, convert_html_to_jpeg


def remove_files():
    # Get a list of all the files in the current directory
    md_files = glob.glob('*.md')
    csv_files = glob.glob('*.csv')
    result_csv_files = glob.glob('result/*.csv')

    all_files = md_files + csv_files + result_csv_files

    if 'readme.md' in all_files:
        all_files.remove('readme.md')

    for filename in all_files:
        os.remove(filename)
        # print(f'Deleted file: {filename}')


def get_api_key():
    try:
        with open('key.txt', 'r') as file:
            api_key = file.read().strip()
        return api_key
    except Exception as e:
        logging.error(f'Failed to read API key: {e}')
        return None
def get_bitly_api_key():
    try:
        with open('bitly.txt', 'r') as file:
            api_key = file.read().strip()
        return api_key
    except Exception as e:
        logging.error(f'Failed to read API key: {e}')
        return None

def get_robots():
    try:
        conn = http.client.HTTPSConnection("api.browse.ai")
        headers = {'Authorization': f"Bearer {get_api_key()}"}
        conn.request("GET", "/v2/robots", headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        logging.error(f'Failed to get robots: {e}')
        return None


def get_robot_tasks(robot_id):
    try:
        conn = http.client.HTTPSConnection("api.browse.ai")
        headers = {'Authorization': f"Bearer {get_api_key()}"}
        conn.request("GET", f"/v2/robots/{robot_id}/tasks?page=1", headers=headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        return json.loads(data)
    except Exception as e:
        logging.error(f'Failed to get robot tasks: {e}')
        return None


def post_robot_task(robot_id, origin_url):
    try:
        url = f"https://api.browse.ai/v2/robots/{robot_id}/tasks"
        payload = {"inputParameters": {"originUrl": origin_url}}
        headers = {"Authorization": f"Bearer {get_api_key()}"}
        response = requests.request("POST", url, json=payload, headers=headers)
        return response.json()
    except Exception as e:
        logging.error(f'Failed to post robot task: {e}')
        return None


def generate_affiliate_link(product_link, affiliate_id):
    link = f"{product_link}/ref=nosim?tag={affiliate_id}"
    # print(link)
    return link


def generate_short_link(affiliate_link, bitly_key):
    url = "https://api-ssl.bitly.com/v4/shorten"
    headers = {
        "Authorization": f"Bearer {bitly_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "long_url": affiliate_link,
        "domain": "bit.ly",
    }
    response = requests.post(url, headers=headers, json=payload)

    # Check if the request was successful
    if response.status_code != 200:
        logging.error("Bitly API request failed. Error: %s", response.text)
        return None

    return response.json()['link']


def clean_url(encoded_url):
    # Decode URL
    decoded_url = urllib.parse.unquote(encoded_url)

    # Extract amazon link from decoded URL
    amazon_url = urllib.parse.urlparse(decoded_url).query.split('url=')[1].split('&')[0]

    # Clean the Amazon URL to remove tracking information
    clean_url = amazon_url.split('?')[0]
    # print(clean_url)
    return clean_url
def temu_url_parser(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    product_link_tag = soup.find('a', {'class': '_3VEjS46S _2IVkRQY-'})
    relative_url = product_link_tag['href'] if product_link_tag else 'No product link found'
    base_url = "https://www.temu.com"
    full_url = base_url + relative_url if relative_url.startswith("/") else relative_url
    return full_url

def download_product_image(image_url, product_number):
    # Create the directory if it doesn't exist
    if not os.path.exists('./product_image'):
        os.makedirs('./product_image')

    # Download the image
    response = requests.get(image_url, stream=True)
    response.raise_for_status()

    # Save the image to the file
    filename = f'./product_image/product_{product_number}.jpeg'
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def parse_prices(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extracting the current price
    current_price_div = soup.find('div', {'class': '_2Ci4uR69', 'aria-hidden': 'true'})
    current_price = float(current_price_div.text) if current_price_div else None

    # Extracting the market price
    market_price_span = soup.find('span', {'data-type': 'marketPrice'})
    market_price = float(market_price_span.text[1:]) if market_price_span else None

    return current_price, market_price
def get_robot_task_items(robot_id):
    res = get_robot_tasks(robot_id)
    print(res)
    tasks_items = []
    for task in res.get('result', {}).get('robotTasks', {}).get('items', []):
        task_items = {}
        task_items['taskId'] = task.get('taskId')
        task_items['items'] = []  # Initialize items list
        if 'capturedLists' in task:
            for item_list_name in ['dealmoon ult2', 'temu']:
                if item_list_name in task['capturedLists']:
                    items = task['capturedLists'][item_list_name]
                    for index, item in enumerate(items, start=1):  # Adding an index here
                        # Parse product name, price, and image
                        product_name = item.get('product name', None)
                        price_html = item.get('price', None)
                        image = item.get('image', None)

                        # Parse price from HTML content
                        if price_html:
                            current_price, market_price = parse_prices(price_html)
                        else:
                            current_price, market_price = None, None

                        # Only call temu_url_parser if image is not None
                        if image:
                            url = temu_url_parser(image)
                        else:
                            url = None
                        current_price = item.get('current price', current_price)
                        market_price = item.get('market price', market_price)

                        task_items['items'].append({
                            'Position': index,  # Store position in item
                            'product_name': product_name,
                            'current_price': current_price,
                            'market_price': market_price,
                            'image': image,
                            'url': url
                        })
        tasks_items.append(task_items)
    return tasks_items

def print_task_items(task_items):
    for task in task_items:
        print(f"Task ID: {task['taskId']}")
        for item in task['items']:
            print(f"{item['Position']}: Name: {item['product_name']}, Current Price: {item['current_price']}, Market Price: {item['market_price']}, URL: {item['url']}\n")

robot_id = '8d798dd1-f3d9-435c-aa33-c9d93f6cb57e'
task_items = get_robot_task_items(robot_id)
print_task_items(task_items)


def print_task_items(task_items):
    for task in task_items:
        print(f"Task ID: {task['taskId']}")
        for item in task['items']:
            print(f"{item['Position']}:Product Name: {item['product_name']}, Current Price: {item['current_price']}, Market Price: {item['market_price']}, URL: {item['url']}\n")

robot_id = '4fc8cf41-ab7e-4d59-8435-c5318485c26f'
task_items = get_robot_task_items(robot_id)
print_task_items(task_items)
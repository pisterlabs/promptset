import json
import time
import openai
from collections import defaultdict
import requests
import validators
from helpers import *
import time
import re
import html
import textwrap
from bs4 import BeautifulSoup
from request_counter import count_requests, global_counter, get
from datetime import datetime
from flask_socketio import SocketIO, emit
from config import Config
import random
import query
from prompts import *

app_settings = {}
category_settings = {}
seo_settings = {}

stop_category_process = {}

formatted_now = None
now = None

def reset_stop():
    print('Resetting the process...')
    global stop_category_process
    stop_category_process = False

socketio = None

def set_socketio(sio):
    global socketio
    socketio = sio



#################### MAIN FUNCTIONS ####################
### GET CATEGORY INFO ###
def getCategoryInfo(app_settings):

    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
    }
    url = f"{app_settings['url']}/api/v2/categories?fields[categories]=parent_id,name,url_handle&page[size]=100"

    if not validators.url(url):
        raise Exception("The URL provided in 'app_settings' is not valid")

    categories_by_id = {}
    children_by_parent_id = defaultdict(list)
    max_retries = 15
    next_url = url

    while next_url:
        for attempt in range(max_retries):
            try:
                response = get(next_url, headers=headers)
                if response.status_code != 200:
                    raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")

                data = response.json()
                for category in data['data']:
                    id = category['id']
                    parent_id = str(category['attributes']['parent_id'])  # Convert parent_id to string here
                    categories_by_id[id] = category
                    if parent_id != 'None':
                        children_by_parent_id[parent_id].append(id)

                next_url = data['links']['next'] + "&fields[categories]=parent_id,name,url_handle" if "next" in data["links"] else None
                break
            except Exception as e:
                if attempt < max_retries - 1:  
                    wait_time = 5 * (attempt + 1)
                    
                    print(f"Error occured at CloudCart. Waiting for {wait_time} seconds before retrying.")
                    time.sleep(wait_time)
                else:  
                    raise

    return categories_by_id, children_by_parent_id

### GET CATEGORY LEVELS ###
def getCategoryLevels(app_settings, category_id, categories_by_id, children_by_parent_id):
    info = {'root_category': None, 'same_level_categories': [], 'sub_level_categories': []}
    category = categories_by_id.get(str(category_id))

    if category is None:
        print(f"Category with ID {category_id} does not exist")
        return info

    root_category_id = str(category['attributes']['parent_id'])

    if root_category_id != 'None':
        root_category = categories_by_id.get(root_category_id, None)
        if root_category is not None:
            info['root_category'] = {'id': root_category['id'], 'name': root_category['attributes']['name'], 'url': app_settings['url'] + "/category/" + root_category['attributes']['url_handle']}

        # same-level categories are those with the same parent as the target category
        same_level_ids = children_by_parent_id.get(root_category_id, [])
    else:
        # if the target category is a root category, then all root categories are its same-level categories
        same_level_ids = [cat_id for cat_id, cat in categories_by_id.items() if str(cat['attributes']['parent_id']) == 'None']

    same_level_ids = [str(id) for id in same_level_ids]
    info['same_level_categories'] = [{'id': categories_by_id[id]['id'], 'name': categories_by_id[id]['attributes']['name'], 'url': app_settings['url'] + "/category/" + categories_by_id[id]['attributes']['url_handle']} for id in same_level_ids if id != str(category['id'])]
    
    sub_level_ids = children_by_parent_id.get(str(category['id']), [])
    sub_level_ids = [str(id) for id in sub_level_ids]
    info['sub_level_categories'] = [{'id': categories_by_id[id]['id'], 'name': categories_by_id[id]['attributes']['name'], 'url': app_settings['url'] + "/category/" + categories_by_id[id]['attributes']['url_handle']} for id in sub_level_ids]

    return info

### GET ORDERED PRODUCTS BY SALES ###
def getOrderedProductsbySales(category_id, app_settings, category_settings, target_category_info):
    headers = {'X-CloudCart-ApiKey': f'{app_settings["X-CloudCart-ApiKey"]}'}
    vendors = getVendors(app_settings)
    vendor_mapping = {str(vendor['id']): {'name': vendor['attributes']['name'], 'url': app_settings['url'] + "/vendor/" + vendor['attributes']['url_handle']} for vendor in vendors}

    ordered_products_url = f'{app_settings["url"]}/api/v2/order-products?filter[category_id]={category_id}&page[size]=100&sort=-order_id'
    ordered_product_data = []
    brand_count = {}  # Initialize brand_count
    for page_number in range(1, int(category_settings['max_order_pages'] + 1)):
        page_url = f'{ordered_products_url}&page[number]={page_number}'
        response = get(page_url, headers=headers)
        data = response.json().get('data', [])
        ordered_product_data += data

    if not ordered_product_data:
        print("No orders found for the category.")
        return {}, []

    product_order_count = {}
    for product in ordered_product_data:
        product_id = str(product['attributes']['product_id'])
        product_order_count[product_id] = product_order_count.get(product_id, 0) + 1

    sorted_products = sorted(product_order_count.items(), key=lambda x: x[1], reverse=True)

    active_products_url = f'{app_settings["url"]}/api/v2/products?filter[category_id]={category_id}&page[size]=100&filter[active]=yes'
    active_product_data = []
    page_number = 1
    
    #while category_settings['only_active_products']:
    while True:
        page_url = f'{active_products_url}&page[number]={page_number}'
        response = get(page_url, headers=headers)
        data = response.json().get('data', [])
        active_product_data += data

        next_page_url = response.json().get('links', {}).get('next')
        if next_page_url is None or not data:
            break

        page_number += 1

    if not active_product_data:
        print("No active products found in the category.")
        return {}, []

    active_product_ids = {str(product['id']) for product in active_product_data}

    # Assigning keys based on 'only_active'
    best_sellers = [product for product in active_product_data if str(product['id']) in active_product_ids]
    price_key = 'price_from'
    url_handle_key = 'url_handle'
    product_id_key = 'id'  # In this case 'id' is used for product_id
    '''
    if category_settings['only_active_products']:
        # For active products
        best_sellers = [product for product in active_product_data if str(product['id']) in active_product_ids]
        price_key = 'price_from'
        url_handle_key = 'url_handle'
        product_id_key = 'id'  # In this case 'id' is used for product_id
    else:
        # For ordered products
        best_sellers = [product for product in ordered_product_data if str(product['attributes']['product_id']) in (product_id for product_id, count in sorted_products)]
        price_key = 'order_price'
        url_handle_key = ''  # Since 'url_handle' is not available in ordered_product_data
        product_id_key = 'product_id'  # In this case 'product_id' is used
    '''

    if best_sellers:
        prices = [product['attributes'][price_key] / 100 for product in best_sellers if product['attributes'][price_key] is not None]
        if prices:
            lowest_price = round(min(prices), 2)
            highest_price = round(max(prices), 2)
        else:
            lowest_price = 0.00
            highest_price = 0.00

    else:
        print("No active best sellers found.")
        return {}, []

    price_range = round((highest_price - lowest_price) / 3, 2)

    products_by_sales = {
        "entry_level_products": [],
        "mid_size_products": [],
        "hi_end_products": []
    }

    for product in best_sellers:
        product_id = str(product[product_id_key])
        '''
        if category_settings['only_active_products']:
            product_id = str(product[product_id_key])  # Use 'id' for active products
        else:
            product_id = str(product['attributes'][product_id_key])  # Use 'product_id' for ordered products
        '''
        price = product['attributes'][price_key] / 100 if product['attributes'][price_key] is not None else 0

        brand_info = vendor_mapping.get(str(product['attributes']['vendor_id']), {'name': str(product['attributes']['vendor_id']), 'url': ''})

        brand_name = brand_info['name']
        brand_count[brand_name] = brand_count.get(brand_name, 0) + 1

        product_url = app_settings['url'] + "/product/" + (product['attributes'][url_handle_key] if url_handle_key else '')

        # extract images from the response.
        image_id = product['attributes'].get('image_id', None)
        # get the image url from this endpoint /api/v2/images/237
        if image_id:
            image_url = fetch_image_url(app_settings['url'], image_id, headers)        
        else:
            image_url = None

        product_info = {
            "name": product['attributes']['name'],
            "orders": product_order_count.get(product_id, 0),
            "price": "{:.2f}".format(price),
            "brand_name": brand_name,
            "product_url": product_url,  # added product_url
            "brand_url": brand_info['url'],  # added brand_url
            "image_url": image_url
        }

        if price <= lowest_price + price_range:
            products_by_sales["entry_level_products"].append(product_info)
        elif price <= lowest_price + 2 * price_range:
            products_by_sales["mid_size_products"].append(product_info)
        else:
            products_by_sales["hi_end_products"].append(product_info)

    sales_limit = category_settings['add_best_selling_products'] if category_settings.get('include_sales_info', False) else 0
    faq_limit = category_settings['add_best_selling_products_faq'] if category_settings.get('include_faq_info', False) else 0
    max_products_limit = max(sales_limit, faq_limit)    
    for category in products_by_sales:
        products_by_sales[category] = sorted(products_by_sales[category], key=lambda x: x["orders"], reverse=True)[:max_products_limit]

    sales_limit_brands = category_settings['add_top_brands'] if category_settings.get('include_sales_info', False) else 0
    faq_limit_brands = category_settings['add_top_brands_faq'] if category_settings.get('include_faq_info', False) else 0
    max_brands_limit = max(sales_limit_brands, faq_limit_brands) 
    best_selling_brands = sorted(brand_count.items(), key=lambda x: x[1], reverse=True)[:max_brands_limit]

    best_selling_brands = [
        {
            "Brand name": brand, 
            "Orders": count, 
            "url": target_category_info['url_handle'] + "?vendors=" + next((v['url'].split("/vendor/")[-1] for k, v in vendor_mapping.items() if v['name'] == brand), '')
        } 
        for brand, count in best_selling_brands
    ]

    return products_by_sales, best_selling_brands

### GET CATEGORY DETAILS ###
def getCategoryDetails(app_settings, fields=[]):
    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
    }
    
    # Convert the fields list to a comma-separated string
    field_string = ','.join(fields)

    # Construct the URL based on the presence of fields
    base_url = f"{app_settings['url']}/api/v2/categories?page[size]=100"
    url = f"{base_url}&fields[categories]={field_string}" if fields else base_url

    max_retries = 15
    next_url = url
    categories_data = []

    while next_url:
        for attempt in range(max_retries):
            try:
                if not validators.url(next_url):
                    raise Exception("The URL provided is not valid")
                
                response = get(next_url, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")

                data = response.json()

                for item in data.get('data', []):
                    category_data = {}
                    category_data['id'] = item.get('id')
                    
                    # Fetching other attributes if they exist inside the 'attributes' dictionary
                    attributes = item.get('attributes', {})
                    for field in fields:
                        if field != 'id':  # We've already fetched the ID
                            category_data[field] = attributes.get(field)

                    categories_data.append(category_data)

                next_url = data['links']['next'] if "next" in data["links"] else None
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Error occurred. Waiting for {wait_time} seconds before retrying.")
                    time.sleep(wait_time)
                else:
                    raise

    return categories_data


#################### HELPERS ####################
### GET VENDOR DETAILS ###
def getVendorDetails(vendor_id, app_settings):
    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
    }

    # Build the base URL
    url = f"{app_settings['url']}/api/v2/vendors/{vendor_id}?page_size=100"

    if not validators.url(url):
        raise Exception("The URL provided is not valid")

    # Send the request
    response = get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")

    vendor_details = response.json().get('data', None)

    return vendor_details

### GET ALL VENDORS ###
def getVendors(app_settings):
    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
    }

    url = f"{app_settings['url']}/api/v2/vendors?fields[vendors]=name,url_handle&page[size]=100"
    max_retries = 15
    next_url = url
    vendors = []

    while next_url:
        for attempt in range(max_retries):
            try:
                if not validators.url(next_url):
                    raise Exception("The URL provided is not valid")
                
                response = get(next_url, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")

                data = response.json()
                vendors.extend(data.get('data', []))

                next_url = data['links']['next'] if "next" in data["links"] else None
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Error occurred. Waiting for {wait_time} seconds before retrying.")
                    time.sleep(wait_time)
                else:
                    raise

    return vendors

### GET TARGET CATEGORY INFO ###
def getTargetCategoryInfo(db, Processed_category, app_settings, category_id, category_settings, seo_settings, project_id, include_description=False):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")

    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
    }
    
    url = f"{app_settings['url']}/api/v2/categories/{category_id}?fields[categories]=name,description,url_handle,properties&include=properties"
    max_retries = 1
    properties = None

    for attempt in range(max_retries):
        try:
            response = get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")
            # Get the data from the response
            data = response.json().get('data', {})
            url = f"{app_settings['url']}/category/{data['attributes'].get('url_handle')}"
            category_name = data['attributes'].get('name')

            if category_settings['include_properties'] or category_settings['include_properties_faq']:
                # Get links from a page by calling the function get_links_from_page only if the category contains properties
                if data['relationships'].get('properties'):
                    
                    Config.socketio.emit('log', {'data': f'{formatted_now}: Getting links from {url}'},room=str(project_id), namespace='/')
                    properties = get_links_from_page(url, app_settings, category_settings, category_name, seo_settings, project_id)

            description = ''
            # compare the description_lenght with the number of characters 
            # get the description, remove HTML tags
            raw_description = re.sub('<[^<]+?>', '', html.unescape(data['attributes'].get('description')))
            # check

            if category_settings.get('description_length') != 0 and category_settings.get('description_length') < len(raw_description):
                Config.socketio.emit('log', {'data': f'{formatted_now}: The description of the category is longer than the threshold target. The process will end...'}, room=str(project_id), namespace='/')
                raise Exception(f"The description of the category is {len(raw_description)} characters which is longer than the threshold {category_settings.get('description_length')} target. The process will end...")

            # Save the processed category
            # This is information about all fields from the database
            # project_id, category_id, category_structure, category_name, category_prompt, category_description, category_faqs, category_keywords, category_custom_keywords
            query.save_processed_category(db, Processed_category, project_id, category_id, category_name=category_name.lower(), category_url=url) 

            
            

            # Get description_length. If it's missing or not positive, use a default value (e.g., 100).
            description_length = category_settings.get('description_length')
            if not description_length or description_length <= 0:
                description_length = 10  # Set default width as 100, change to a suitable value as per your needs

            # truncate to the end of the sentence at or before description_length characters
            truncated_description = textwrap.shorten(raw_description, width=description_length, placeholder="...")
            
            if include_description:
                description = data['attributes'].get('description'),
            
            # format the output
            target_category_info = {
                "name": data['attributes'].get('name'),
                "id": data.get('id'),
                "url_handle": url,
                "description": description,
                # add condition to check if the category contains properties
                "properties": properties if properties else []
            }
            
            # If request was successful, break out of the loop and return category info
            return target_category_info
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                Config.socketio.emit('log', {'data': f'{formatted_now}: Error occurred. Waiting for {wait_time} seconds before retrying.'},room=str(project_id), namespace='/')

                time.sleep(wait_time)
            else:
                raise

### SCRAPE PROPERTIES FROM THE CATEGORY PAGE ###
def get_links_from_page(url, app_settings, category_settings, category_name, seo_settings, project_id):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    Config.socketio.emit('log', {'data': f'{formatted_now}: Getting properties from {url}'},room=str(project_id), namespace='/')
    response = get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # List of possible selectors
    selectors = [
        '._filter-category-property',
        '._filter-category-properties ._filter-category-property',
        '._filter-category-property .js-section-title js-filter-category-property-toggle active',
        '._filter-category-property'
    ]

    # Try each selector until we find results
    for selector in selectors:
        property_blocks = soup.select(selector)
        if property_blocks:
            break

    results = {}
    prop_count = 0  # Counter for properties
    max_props = int(category_settings.get('max_props', 0))  # Maximum number of properties

    for block in property_blocks:
        if prop_count >= max_props:
            break  # Stop the loop if we have reached the maximum number of properties

        property_title = block.select_one('._filter-category-property-title h5').text
        form_rows = block.select('._form-row')
        count = 0 

        if property_title not in results:
            results[property_title] = []
            prop_count += 1  # Increment the property counter only when a new property is encountered
        else:
            continue  # Skip to the next block if the property is already in results
        if category_settings['include_properties'] or category_settings['include_properties_faq']:
            for row in form_rows:
                if count < int(category_settings.get('max_property_values', 0)) or count < int(category_settings.get('max_property_values_faq', 0)):
                    input_tag = row.select_one('input.category-property-filter')
                    label = row.select_one('label._checkbox')

                    property_name = input_tag.get('data-property-name') if input_tag else None
                    value = input_tag.get('value') if input_tag else None
                    value_title = label.text.strip() if label else ''

                    if property_name and value:
                        new_url = f'{url}?{property_name}={value}'

                        results[property_title].append([value_title, new_url])

                    count += 1
    Config.socketio.emit('log', {'data': f'{formatted_now}: Found properties...'},room=str(project_id), namespace='/')
    if results:
        if seo_settings.get('generate_keywords'):
            keywords = craft_keywords_urls(app_settings, category_settings, seo_settings, category_name, results, project_id)
            keywords = keywords.strip()
            return json.loads(keywords)
        else:
            return results

### GET CATEGORY IDS only ###
def getCategoryIds(app_settings):
    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
    }

    url = f"{app_settings['url']}/api/v2/categories?fields[categories]=id&page[size]=100"
    max_retries = 15
    next_url = url
    category_ids = []

    while next_url:
        for attempt in range(max_retries):
            try:
                if not validators.url(next_url):
                    raise Exception("The URL provided is not valid")
                
                response = get(next_url, headers=headers)

                if response.status_code != 200:
                    raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")

                data = response.json()
                category_ids.extend([category.get('id') for category in data.get('data', [])])

                next_url = data['links']['next'] if "next" in data["links"] else None
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"Error occurred. Waiting for {wait_time} seconds before retrying.")
                    time.sleep(wait_time)
                else:
                    raise

    return category_ids

#################### OPENAI FUNCTIONS ####################

### CRAFT KEYWORDS FOR EACH PROPERTY VALUE ###
def craft_keywords_urls(app_settings, category_settings, seo_settings, category_name, results, project_id):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
    openai.api_key = app_settings['openai_key']

    if results is None:
        Config.socketio.emit('log', {'data': f'{formatted_now}: No properties found for the category'},room=str(project_id), namespace='/')
        return
    
    json_structure_example = {"season": [["summer", "https://shop.mdl.bg/category/bodi-women?y=1"], ["winter", "https://shop.mdl.bg/category/bodi-women?y=2"]]}
    json_structure_example_str = str(json_structure_example)


    # Modify the prompt to ask for keywords for the specific property value
    prompt = (f"As a {app_settings['language']} SEO researcher working for {category_name} category. Craft maximum of {seo_settings['max_keywords']} SEO-optimized keywords for each property name and its values. Use this {category_name} category name for a context when crafting the keywords and also \"price\" or \"affordable price\". This is an example of the array: {json_structure_example_str} where \"season\" is the name of the property group, \"summer\" is the value of the property and the link is the link to the value. To craft a keyword, use the property name \"season\" as a context and use each values \"summer\" and \"winter\" to generate keywords. For example, use the property name as a context to generate 2 keywords for the value \"summer\" and 2 keywords for the value \"winter\". *** This is the actual information that you need to use to generate keywords: {results} ***")
    #prompt = (f"As a Bulgarian SEO researcher, craft at least {category_settings['max_keywords']} SEO optimized keywords for the property '{property_name}' with value '{value}' in the category '{category_name}'. ***Do not include only the property value, but also additional related keywords.***")

    if app_settings['print_prompt']:
        Config.socketio.emit('log', {'data': f'{formatted_now}: Prompt for keywords generation:\n{prompt}'},room=str(project_id), namespace='/')
        return(prompt)
    
    system_prompt = (
        'The output must be strictly valid JSON structure like this example: '
        '{"y":[{"id":"1","url":"https://shop.mdl.bg/category/bodi-women?y=1","keywords":["keyword1","keyword 2"]},'
        '{"id":"2","url":"https://shop.mdl.bg/category/bodi-women?y=2","keywords":["keyword1","keyword 2"]}]}' 
        f'*** The output must be ONLY the JSON structure explicitly and you should keep the {app_settings["language"]} language from the prompt!***'
    )

    max_retries = 15
    Config.socketio.emit('log', {'data': f'{formatted_now}: Generating keywords...'},room=str(project_id), namespace='/')

    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=app_settings['seo_model'],
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": system_prompt},
                    ],
                temperature=app_settings['temperature'],
            )
            # If the request was successful, break out of the loop
            break
        except openai.error.AuthenticationError:
            # Handle authentication errors (e.g., invalid API key)
            Config.socketio.emit('log', {'data': 'Authentication Error. Check your OpenAI API Key!'}, room=str(1), namespace='/')
            break
        except (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:  
            # Handle APIError, Timeout, and ServiceUnavailableError for retry
            wait_time = 2 * (attempt + 1)
            Config.socketio.emit('log', {'data': f'{formatted_now}: Encountered an issue: {e.error}. Waiting for {wait_time} seconds before retrying.'},room=str(project_id), namespace='/')
            time.sleep(wait_time)
        except Exception as e:
            # Handle all other exceptions without retrying
            Config.socketio.emit('log', {'data': f'{formatted_now}: {e}'},room=str(project_id), namespace='/')
            break
    else:
        raise Exception("Maximum number of retries exceeded.")

    answer = response['choices'][0]['message']['content']

    results = answer
    return results



### CRAFT FAQS ###
def craft_faqs(db, Processed_category, Category_Settings, app_settings, category_settings, seo_settings, results, project_id):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
    Config.socketio.emit('log', {'data': f'{formatted_now}: Crafting FAQs...'},room=str(project_id), namespace='/')
    ### STOP PROCESS IF USER CLICKS STOP ###
    if stop_category_process.get(project_id, False):
        stop_category(project_id)  # Stop process
        Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'},room=str(project_id), namespace='/')
        project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
        if project:
            project.in_progress = False
            db.session.commit()
        return 
    openai.api_key = app_settings['openai_key']
    
    # Extract target_category_info
    target_category_info = results['target_category_info']
    target_category_name = target_category_info['name']
    target_category_url = target_category_info['url_handle']
    target_category_properties = target_category_info['properties']

    prop_output = ''

    if category_settings['include_properties_faq']:
        if isinstance(target_category_properties, dict):
            for i, (prop, values) in enumerate(target_category_properties.items()):
                # If we have already processed the maximum number of properties, break the loop
                if i >= category_settings["max_property_values_faq"]:
                    break

                prop_output += f'Property name "{prop}", '
                for val in values:
                    # Check the type of val
                    if isinstance(val, list):
                        id_ = val[0]
                        url_ = val[1]  # assuming the second item in the list is always a URL
                        prop_output += f'Value {id_}, Link: {url_}, '
                    elif isinstance(val, dict):
                        id_ = val["id"]
                        url_ = val["url"]
                        keywords = val["keywords"]
                        prop_output += f'Keywords: {", ".join(keywords)}, Links: {url_}, '

        elif isinstance(target_category_properties, list):
            # handle the list case here, for example:
            for prop in target_category_properties:
                prop_output += f'Property: {prop}, '

        else:
            raise ValueError("Unexpected type for target_category_properties")

        

    # Extract category_info
    category_info = results.get('category_info')

    if category_info is None:
        top_level_category = []
        same_level_categories = []
        sub_level_categories = []
    else:
        # Same level categories information
        top_level_category_info = category_info.get('root_category')
        if top_level_category_info and 'name' in top_level_category_info and 'url' in top_level_category_info:
            top_level_category = [(top_level_category_info['name'], top_level_category_info['url'])]
        else:
            top_level_category = []

        same_level_categories_info = category_info.get('same_level_categories', [])
        same_level_categories = [(cat['name'], cat['url']) for cat in same_level_categories_info 
                                if cat and 'name' in cat and 'url' in cat]

        # Sub level categories information
        sub_level_categories_info = category_info.get('sub_level_categories', [])
        sub_level_categories = [(cat['name'], cat['url']) for cat in sub_level_categories_info 
                                if cat and 'name' in cat and 'url' in cat]


    if category_settings['include_faq_info']:
        # Extract product level information
        products_info = results.get('products_by_sales', {})

        entry_level_products = []
        mid_size_products = []
        hi_end_products = []

        if 'entry_level_products' in products_info:
            entry_level_products = [(prod['name'], prod['product_url'], prod['price']) 
                                    for prod in products_info['entry_level_products'] 
                                    if 'name' in prod and 'product_url' in prod and 'price' in prod]

        if 'mid_size_products' in products_info:
            mid_size_products = [(prod['name'], prod['product_url'], prod['price']) 
                                for prod in products_info['mid_size_products'] 
                                if 'name' in prod and 'product_url' in prod and 'price' in prod]

        if 'hi_end_products' in products_info:
            hi_end_products = [(prod['name'], prod['product_url'], prod['price']) 
                            for prod in products_info['hi_end_products'] 
                            if 'name' in prod and 'product_url' in prod and 'price' in prod]


    top_brands = []
    if category_settings['add_top_brands_faq']:
        # Extract top brands information
        top_brands = [(brand['Brand name'], brand['url']) for brand in results.get('best_selling_brands', [])]
    

    faq_schema = '<div itemscope itemprop="mainEntity" itemtype="https://schema.org/Question"><h3 itemprop="name"><strong> ***QUESTION-PLACEHOLDER*** </strong></h3><p itemscope itemprop="acceptedAnswer" itemtype="https://schema.org/Answer"><span itemprop="text"> ***ANSWER-PLACEHOLDER*** </span></p></div>'

    ### PROMPTS ###

    

    if category_settings['faq_use_schema']:
        sys_prompt = f'Craft an FAQs of the category in this strictly and valid format: {faq_schema}. The answer for each FAQ must be put at ***ANSWER-PLACEHOLDER***! Additionally: The text at the ***ANSWER-PLACEHOLDER*** must be with appropriate HTML tags to improve its structure. For new lines use "br", for bold: "strong". When you have listings use "ul" and "li" or "ol" for numbers, while preserving the existing tags. Emphasize headings, subheadings, and key points by new lines, and ensure the content flows coherently. DO NOT MENTION ANYTHING FROM THE PROMPT IN ANY CASE!'
    else:
        sys_prompt = f'Craft an FAQs of the category. It have to be written in {app_settings["language"]}. For headings use H3. Make the text readable and for each text block you must add <p> tag. For new lines use "br", for bold: "strong". When you have listings use "ul" and "li" or "ol" for numbers, while preserving the existing tags. Emphasize headings, subheadings, and key points by new lines, and ensure the content flows coherently. DO NOT MENTION ANYTHING FROM THE PROMPT IN ANY CASE!'

    system_prompt = (
        sys_prompt
        )

    
    #### KEYWRDS RESEARCH ####
    
    Config.socketio.emit('log', {'data': f'{formatted_now}: Searching Google for related searches for {target_category_name}'},room=str(project_id), namespace='/')
    cluster_keywords_dict = keywords_one_level(db, Category_Settings, app_settings, main_query=target_category_name, project_id=project_id)


    # Get the keywords from cluster_keywords_dict and convert them into a list
    cluster_keywords_list = cluster_keywords_dict.get(target_category_name, '').split(',')

    # append category_settings['use_main_keywords'] into the main_query list including the keywords from cluster_keywords_list
    if category_settings['use_main_keywords']:
        # Convert the string of keywords into a list
        main_keywords_list = category_settings['use_main_keywords'].split(",")  # assuming keywords are comma-separated
        cluster_keywords_list.extend(main_keywords_list)

    # Remove empty strings from the list and deduplicate
    cluster_keywords_list = list(set([keyword.strip() for keyword in cluster_keywords_list if keyword.strip()]))

    # Convert the list back to a comma-separated string and update the dictionary
    cluster_keywords_dict[target_category_name] = ','.join(cluster_keywords_list)

    cluster_keywords = cluster_keywords_dict

    category_id = target_category_info['id']


    ### Unique keywords ###
    # Extracting the keys and joining them with commas
    top_keywords = ', '.join(cluster_keywords.values())

    # Creating the second string
    key_value_string = ', '.join([f"{key}, {value}" for key, value in cluster_keywords.items()])
    keywords_list = key_value_string.split(', ')
    unique_keywords_list = list(set(keywords_list))
    unique_keywords_string = ', '.join(unique_keywords_list)


    # Introduce faq_count
    faq_count = category_settings['add_faq']

    # Initialize an empty list to store each individual FAQ
    all_faqs = []
    finished_faqs = []
    # Surround FAQ generation with the desired div
    faq_wrapper_start = '<div itemscope itemtype="https://schema.org/FAQPage">'
    faq_wrapper_end = '</div>'

    # Add counter for current iteration
    iteration_counter = 0
    # Extract all keywords from the dictionary's values outside the loop
    
    additional_questions = category_settings['additional_instructions_faq']
    all_keywords = ','.join(cluster_keywords.values()).split(',')
    if additional_questions:
        all_keywords += additional_questions.split(',')
    num_keywords = len(all_keywords)

    Config.socketio.emit('log', {'data': f'{formatted_now}: Preparation for FAQs is ready. {num_keywords} FAQs will be generated'},room=str(project_id), namespace='/')

    # Ensure the loop doesn't exceed the number of keywords available
    loop_limit = num_keywords
    all_keyphrases = []
    for i in range(loop_limit):

        ### STOP PROCESS IF USER CLICKS STOP ###
        if stop_category_process.get(project_id, False):
            stop_category(project_id)  # Stop process
            Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'}, room=str(project_id), namespace='/')
            project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
            if project:
                project.in_progress = False
                db.session.commit()
            return 

        iteration_counter += 1  # Increment the counter
        Config.socketio.emit('log', {'data': f'{formatted_now}: Working on FAQ #{iteration_counter}'}, room=str(project_id), namespace='/')

        # Directly set the current_keyword using the index i
        current_keyword = all_keywords[i]
        
        # Call the function to get related searches for the current_keyword
        Config.socketio.emit('log', {'data': f'{formatted_now}: Searching Google for related searches for {current_keyword}'},room=str(project_id), namespace='/')
        keyphrases_dict = keywords_one_level(db, Category_Settings, app_settings, main_query=current_keyword, project_id=project_id)

        # Extract keywords from the dictionary's values and create a list
        keyphrases_list = ','.join(keyphrases_dict.values()).split(',')
        all_keyphrases.append(keyphrases_list)

        # Here, you can continue with the rest of your logic using current_keyword and keyphrases_list

        prompt = ''
        ### INTRO ###
        prompt += f"*** GENERAL FAQ INSTRUCTIONS: ***\n"
        prompt += f"I want you to act as a proficient SEO content writer for FAQs in {app_settings['language']}"

        prompt += f"Each answer MUST contains minimum {category_settings['faq_length']} words.\n"

        prompt += f"Craft a 100% unique, SEO-optimized question and answer for the FAQ section in {app_settings['language']}. "
        ### INSTRUCTIONS FOR QUESTIONS ###
        prompt += f"INSTRUCTIONS FOR CRAFTING QUESTIONS:\n"
        prompt += f"This is question #{iteration_counter}. Craft a question from this keyword: '{current_keyword}' for the category: '{target_category_name}' with H3 tag.\n"



        if category_settings["include_category_info_faq"]:
            temp_prompt = ""

            # Check if same_level_categories has content
            if same_level_categories:
                temp_prompt += f"Craft questions related with the same level categories: {same_level_categories} "

            # Check if sub_level_categories is not empty
            if sub_level_categories:
                temp_prompt += f"and sub-categories with their links to their category page: {sub_level_categories} "

            if temp_prompt:  # Only add the keyword instruction if temp_prompt has content
                temp_prompt += f"as an SEO expert with a focus on keyword stemming, you must derive and choose the most appropriate stemmed variations from the keywords provided in this list: '{keyphrases_list}'. Understand the core meaning and concept of each keyword, and then incorporate these stemmed variations intuitively and naturally across the entire text.\n"

            # Append or concatenate temp_prompt to the main prompt
            prompt += temp_prompt
        
        ### INSTRUCTIONS FOR ANSWERS ###
        prompt += f"*** INSTRUCTIONS FOR CRAFTING ANSWERS: ***\n"
        # Unique keywords
        
        #prompt += f"To craft an answer, it is important and mandatory to use the following keyphrases: '{keyphrases_list}'."
        prompt += f"As an SEO expert it is important and mandatory to focus on keyword stemming, you must derive and choose the most appropriate stemmed variations from the keywords provided in this list: '{keyphrases_list}'.  Understand the core meaning and concept of each keyword, and then incorporate these stemmed variations intuitively and naturally across the entire text."
        prompt += f"This keywords are related to the main keyword {current_keyword} that the question is crafted from. You must include it as well!\n"
        if prop_output:
            prompt += f"in combination with this category properties: {prop_output}. It is important to communicate the primary features and functions that these products typically have in combination with the keyphrases provided. \n"
        prompt += f"It is important to make sure you bold added keyphrases for more visibility. That way your FAQs will be SEO optimized.\n"

        # Products information
        #if products_info:
        if category_settings['include_faq_info']:
            prompt += f"When the question is related with sales, pricing, how cheap the good is or anything related. The key is to craft a pre sales or post sales question that will help customers to make an informed decision based on their specific needs and preferences. "

            product_info = []

            # Check if entry_level_products has content
            if entry_level_products:
                product_info.append(f"For entry level products: {entry_level_products}")

            # Check if mid_size_products has content
            if mid_size_products:
                product_info.append(f"middle size products: {mid_size_products}")

            # Check if hi_end_products has content
            if hi_end_products:
                product_info.append(f"and the high-end (flagman) products: {hi_end_products}")

            # If there's product information available, construct the prompt
            if product_info:
                temp_prompt = "Craft a question related with the best products or pricing ranges and use the following information for product levels to highlight the best products for each category level. "
                
                # Join the product information together with commas and spaces
                product_info_str = ', '.join(product_info)

                # Append or concatenate product_info_str to the temp prompt
                temp_prompt += product_info_str + ". *** DO NOT MENTION EXACT PRICES ***\n"
                
                # Append or concatenate temp_prompt to the main prompt
                prompt += temp_prompt
                prompt += f"Format the list of the products in visible and readable way. Make sure to link the products.\n"

            
        # Top brands information
        has_top_brands_info = (isinstance(top_brands, list) and len(top_brands) > 0) or \
            (isinstance(top_brands, str) and top_brands.strip() != "")

        if has_top_brands_info:
            if category_settings["add_top_brands_faq"] and not category_settings['faq_top_brands_links']:
                prompt += f"Highlighting the Top Brands at the FAQs: {top_brands}, mentioning unique selling points of each in few lines. Do not link each brand. Do not use the URLs.\n\n"
            elif category_settings["add_top_brands_faq"] and category_settings['faq_top_brands_links']:
                prompt += f"7. Highlighting the Top Brands at the FAQs: {top_brands}, mentioning unique selling points of each in few lines. Make a link of each brand.\n"

        # Categpry related information
        if category_settings["include_category_info_faq"]:
            temp_prompt = ""
            
            # Check if same_level_categories has content
            if same_level_categories:
                temp_prompt += f"At your answer add information related with the same level categories: {same_level_categories} "

            # Check if sub_level_categories is not empty
            if sub_level_categories:
                temp_prompt += f"and sub-categories: {sub_level_categories} by adding links to their category page"

            if temp_prompt:  # Only add the keyword instruction if temp_prompt has content
                temp_prompt += f"focus on keyword stemming, you must derive and choose the most appropriate stemmed variations from the keywords provided in this list: '{current_keyword}, {keyphrases_list}'. . Understand the core meaning and concept of each keyword, and then incorporate these stemmed variations intuitively and naturally across the entire FAQ.\n"
            # Append or concatenate temp_prompt to the main prompt
            prompt += temp_prompt

        
        
        ### GENERAL CONCLUSION INSTRUCTIONS ###
        prompt += f"*** FINAL INSTRUCTIONS: ***\n"
        prompt += f"Each Question and Answer should be written in your own words, without copying from other sources and must use provided keyphrases in {app_settings['language']}\n"
        prompt += f"Utilize an informal tone, personal pronouns, active voice, rhetorical questions, and incorporate analogies and metaphors. Keep the text simple, brief and very well formated.\n"
        
        #prompt += "IMPORTANT!: YOU MUST to format the given answer text with appropriate HTML tags to make it well-organized and visually appealing. IT IS A MUST"
        prompt += "DO NOT ADD OR REFFER TO ANY WORDS OF THE THIS PROMPT! ALL OF THE INFORMATION IS FOR BUILDING THE PERFECT FAQ section! DO NOT MENTION ANYTHING FROM THE PROMPT IN ANY CASE!\n"


        if app_settings['print_prompt']:
            Config.socketio.emit('log', {'data': f'{formatted_now}: FAQ System prompt:\n{system_prompt}'},room=str(project_id), namespace='/')
            Config.socketio.emit('log', {'data': f'{formatted_now}: FAQ Prompt:\n{prompt}'},room=str(project_id), namespace='/')
            return
        ### STOP PROCESS IF USER CLICKS STOP ###
        if stop_category_process.get(project_id, False):
            stop_category(project_id)  # Stop process
            Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'},room=str(project_id), namespace='/')
            project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
            if project:
                project.in_progress = False
                db.session.commit()
            return 
        
        Config.socketio.emit('log', {'data': f'{formatted_now}: Content creation for FAQ #{iteration_counter}'},room=str(project_id), namespace='/')

        for attempt in range(15):
            try:
                response = openai.ChatCompletion.create(
                    model=app_settings['model'],
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "system", "content": system_prompt},
                    ],
                    temperature=0,
                )
                # If the request was successful, append FAQ to all_faqs and break out of the loop
                all_faqs.append(response['choices'][0]['message']['content'])
                break
            except openai.error.AuthenticationError:
                # Handle authentication errors (e.g., invalid API key)
                Config.socketio.emit('log', {'data': 'Authentication Error. Check your OpenAI API Key!'}, room=str(1), namespace='/')
                break
            except (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError) as e:  
                # Handle APIError, Timeout, and ServiceUnavailableError for retry
                wait_time = 2 * (attempt + 1)
                Config.socketio.emit('log', {'data': f'{formatted_now}: Encountered an issue: {e.error}. Waiting for {wait_time} seconds before retrying.'},room=str(project_id), namespace='/')
                time.sleep(wait_time)
            except Exception as e:
                # Handle all other exceptions without retrying
                Config.socketio.emit('log', {'data': f'{formatted_now}: {e}'},room=str(project_id), namespace='/')
                break
        else:
            raise Exception("Maximum number of retries exceeded.")



    #### KEYWORDS CLEANUP ####    
    all_faq_keywords = [item for sublist in all_keyphrases for item in sublist] + all_keywords
    # Remove duplicates
    all_faq_keywords = list(set(all_faq_keywords))
    unique_all_faq_keywords_string = ', '.join(all_faq_keywords)

    ### Save to the db the cluster keywords ###
    import query
    #Get keywords from the Processed_category table from column category_keywords and merge them with the new keywords
    category_keywords_records = db.session.query(Processed_category).filter_by(project_id=project_id, category_id=category_id).first()

    if category_keywords_records.category_keywords:
        category_keywords_list = category_keywords_records.category_keywords.split(', ')
        category_keywords_list = [keyword.lower().strip() for keyword in category_keywords_list]

        unique_keywords_list = unique_all_faq_keywords_string.split(', ')
        unique_keywords_list = [keyword.lower().strip() for keyword in unique_keywords_list]

        distinct_keywords_set = set(category_keywords_list + unique_keywords_list)
        distinct_keywords_list = list(distinct_keywords_set)
        category_keywords = ', '.join(distinct_keywords_list)
        
        query.save_processed_category(db, Processed_category, project_id, category_id, category_keywords=category_keywords)
    else:
        query.save_processed_category(db, Processed_category, project_id, category_id, category_keywords=unique_all_faq_keywords_string)

    ##########################


        

    # Combine all FAQs and wrap them with the FAQ div
    if category_settings['faq_include_category_name_at_headings']:
        prompt = f"Craft FAQ section heading in {app_settings['language']} language for the category: {target_category_name}"
    else:
        prompt = f"Craft FAQ section heading in {app_settings['language']}"
    system_prompt = f"Return just the heading. The language is {app_settings['language']}.  *** DO NOT MENTION ANY INSTRUCTIONS FROM YOUR PROMPT! ***"
    faq_heading = openai_generic(app_settings, prompt, system_prompt)




    heading = f"<h2>{faq_heading}</h2>"
    all_faqs.insert(0, heading)

    # Adding Wikipedia link at the bottom of the FAQs
    translated_category_name = google_translate(f"{target_category_name}", "en")

    if category_settings['faq_include_category_name_at_headings'] and category_settings['faq_wiki_link_authority']:
        
        #Config.socketio.emit('log', {'data': f'{formatted_now}: Building wikipedia link for category...'}, room=str(project_id), namespace='/')
        #query = f"{target_category_name} wikipedia"
        Config.socketio.emit('log', {'data': f'{formatted_now}: Searching Wikipedia for {target_category_name}...'}, room=str(project_id), namespace='/')
        #wikilink = google_custom_search(query)
        wikilink = get_wikipedia_url(target_category_name, lang='bg')
        if wikilink:
            Config.socketio.emit('log', {'data': f'{formatted_now}: Adding Wikipedia link: {wikilink}'}, room=str(project_id), namespace='/')
            wikipedia_link = f'<a href="{wikilink}" target="_blank">read more at Wikipedia</a>'
        else:
            Config.socketio.emit('log', {'data': f'{formatted_now}: No results found at WIKIPEDIA BG for {target_category_name}. Searching english version...'}, room=str(project_id), namespace='/')
            Config.socketio.emit('log', {'data': f'{formatted_now}: Searching Wikipedia for {translated_category_name}...'}, room=str(project_id), namespace='/')

            wikilink = get_wikipedia_url(translated_category_name, lang='en')
            if wikilink:
                Config.socketio.emit('log', {'data': f'{formatted_now}: Adding Wikipedia link: {wikilink}'}, room=str(project_id), namespace='/')
                wikipedia_link = f'<a href="{wikilink}" target="_blank">read more at Wikipedia</a>'
            else:
                Config.socketio.emit('log', {'data': f'{formatted_now}: No results found for {translated_category_name} at WIKIPEDIA EN. Searching Google...'}, room=str(project_id), namespace='/')
                query = f"{translated_category_name} wikipedia"
                wikilink = google_custom_search(query)
                Config.socketio.emit('log', {'data': f'{formatted_now}: Validating the link...'}, room=str(project_id), namespace='/')
                if is_valid_url(wikilink):  # Validate the URL
                    Config.socketio.emit('log', {'data': f'{formatted_now}: Adding Wikipedia link: {wikilink}'}, room=str(project_id), namespace='/')
                    wikipedia_link = f'<a href="{wikilink}" target="_blank">read more at Wikipedia</a>'
                else:
                    Config.socketio.emit('log', {'data': f'{formatted_now}: The link is invalid: {wikilink}'}, room=str(project_id), namespace='/')
                    wikipedia_link = target_category_name  # Use plain text if the URL is not valid
        
        prompt = f"Add a link to the Wikipedia page: {wikipedia_link} to read more about {target_category_name}\n"
        system_prompt = f"The language must be {app_settings['language']}.  *** DO NOT MENTION ANY INSTRUCTIONS FROM YOUR PROMPT! ***"
        read_more_at_wiki = openai_generic(app_settings, prompt, system_prompt)
        read_more = f"<p>{read_more_at_wiki}</p>"
        all_faqs.append(read_more)


    
    

    # Action 1: Make a link to the brand
    if len(top_brands) > 0 and category_settings['faq_top_brands_links']:
        Config.socketio.emit('log', {'data': f'{formatted_now}: Building brand link for FAQs...'}, room=str(project_id), namespace='/')
        
        brand = random.choice(top_brands)[0]
        query_brand = f"{translated_category_name} {brand} official website"
        Config.socketio.emit('log', {'data': f'{formatted_now}: Searching Google for trusted website for {brand}'}, room=str(project_id), namespace='/')
        brand_link = google_custom_search(query_brand)
        
        Config.socketio.emit('log', {'data': f'{formatted_now}: Validating the link...'}, room=str(project_id), namespace='/')
        if is_valid_url(brand_link):  # Validate the URL
            linked_brand = f'<a href="{brand_link}">{brand}</a>'
        else:
            linked_brand = brand  # Use plain text if the URL is not valid

        # Replace brand name in all_faqs
        for idx, faq in enumerate(all_faqs):
            if brand in faq:
                all_faqs[idx] = faq.replace(brand, linked_brand, 1)


    
    if category_settings['faq_use_schema']:
        finished_faqs = faq_wrapper_start + ''.join(all_faqs) + faq_wrapper_end
    else:
        finished_faqs = ''.join(all_faqs)
    return finished_faqs

 
### CRAFT DESCRIPTION FOR THE CATEGORY ###
def generate_category_description(db, Category_Settings, Processed_category, app_settings, category_settings, seo_settings, results, project_id):
    openai.api_key = app_settings['openai_key']

    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")

    if results is None:
        Config.socketio.emit('log', {'data': f'{formatted_now}: No properties found for the category.'},room=str(project_id), namespace='/')
        return


    ###################################
    ### DEFINIG IMPORTANT VARIABLES ###
    ###################################

    target_category_info = results['target_category_info']
    target_category_name = target_category_info['name']
    target_category_url = target_category_info['url_handle']
    target_category_properties = target_category_info['properties']

    prop_output = ''

    if category_settings['include_properties']:
        if isinstance(target_category_properties, dict):
            for i, (prop, values) in enumerate(target_category_properties.items()):
                # If we have already processed the maximum number of properties, break the loop
                if i >= category_settings["max_property_values"]:
                    break

                prop_output += f'Property name "{prop}", '
                for val in values:
                    # Check the type of val
                    if isinstance(val, list):
                        id_ = val[0]
                        url_ = val[1]  # assuming the second item in the list is always a URL
                        prop_output += f'Value {id_}, Link: {url_}, '
                    elif isinstance(val, dict):
                        id_ = val["id"]
                        url_ = val["url"]
                        keywords = val["keywords"]
                        prop_output += f'Keywords: {", ".join(keywords)}, Links: {url_}, '

        elif isinstance(target_category_properties, list):
            # handle the list case here, for example:
            for prop in target_category_properties:
                prop_output += f'Property: {prop}, '

        else:
            raise ValueError("Unexpected type for target_category_properties")

    top_level_category = []
    same_level_categories = []
    sub_level_categories = []

    category_info = results.get('category_info')

    if category_info is None:
        top_level_category = []
        same_level_categories = []
        sub_level_categories = []
    else:
        # Same level categories information
        top_level_category_info = category_info.get('root_category')
        if top_level_category_info and 'name' in top_level_category_info and 'url' in top_level_category_info:
            top_level_category = [(top_level_category_info['name'], top_level_category_info['url'])]
        else:
            top_level_category = []

        same_level_categories_info = category_info.get('same_level_categories', [])
        same_level_categories = [(cat['name'], cat['url']) for cat in same_level_categories_info 
                                if cat and 'name' in cat and 'url' in cat]

        # Sub level categories information
        sub_level_categories_info = category_info.get('sub_level_categories', [])
        sub_level_categories = [(cat['name'], cat['url']) for cat in sub_level_categories_info 
                                if cat and 'name' in cat and 'url' in cat]

    entry_level_products = []
    mid_size_products = []
    hi_end_products = []


    if category_settings['include_sales_info']:
        products_info = results.get('products_by_sales', {})

        entry_level_products = []
        mid_size_products = []
        hi_end_products = []

        if 'entry_level_products' in products_info:
            entry_level_products = [(prod['name'], prod['product_url'], prod['price'], prod['image_url']) 
                                    for prod in products_info['entry_level_products'] 
                                    if 'name' in prod and 'product_url' in prod and 'price' in prod and 'image_url' in prod]

        if 'mid_size_products' in products_info:
            mid_size_products = [(prod['name'], prod['product_url'], prod['price'], prod['image_url']) 
                                for prod in products_info['mid_size_products'] 
                                if 'name' in prod and 'product_url' in prod and 'price' in prod and 'image_url' in prod]

        if 'hi_end_products' in products_info:
            hi_end_products = [(prod['name'], prod['product_url'], prod['price'], prod['image_url']) 
                            for prod in products_info['hi_end_products'] 
                            if 'name' in prod and 'product_url' in prod and 'price' in prod and 'image_url' in prod]


    
    top_brands = []
    if category_settings['add_top_brands']:
        top_brands = [(brand['Brand name'], brand['url']) for brand in results.get('best_selling_brands', [])]
    

    ### STOP PROCESS IF USER CLICKS STOP ###
    if stop_category_process.get(project_id, False):
        stop_category(project_id)  # Stop process
        Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'},room=str(project_id), namespace='/')
        project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
        if project:
            project.in_progress = False
            db.session.commit()
        return 
    
    # Check if top_brands contains meaningful information
    has_top_brands_info = (isinstance(top_brands, list) and len(top_brands) > 0) or \
                        (isinstance(top_brands, str) and top_brands.strip() != "")


    ##########################
    #### KEYWRDS RESEARCH ####
    ##########################

    # Check DB Processed categories for the cluster keywords for the target category and if they exist, use them else search for them
    #cluster_keywords_dict = query.get_processed_category(db, Processed_category, project_id, target_category_info['id'])
    #if cluster_keywords_dict is None:
    
    category_id = target_category_info['id']
    unique_keywords_string = ''
    

    if seo_settings['generic_keywords']:

        # Check if unique_keywords_string is empty from the DB
        category_keywords_records = db.session.query(Processed_category).filter_by(project_id=project_id, category_id=category_id).first()
        if category_keywords_records.category_keywords:
            unique_keywords_string = category_keywords_records.category_keywords
        else:
            Config.socketio.emit('log', {'data': f'{formatted_now}: Searching Google for related searches for {target_category_name}'},room=str(project_id), namespace='/')
            cluster_keywords_dict = keyword_subsequence(db, Category_Settings, app_settings, main_query=target_category_name, project_id=project_id)

            # Process the result to ensure it matches the format of keywords_one_level
            for key, value in cluster_keywords_dict.items():
                if isinstance(value, dict) and 'error' in value:
                    cluster_keywords_dict[key] = "error: " + value['error']
                elif isinstance(value, list):
                    cluster_keywords_dict[key] = ', '.join(value)

            # Get the keywords from cluster_keywords_dict and convert them into a list
            cluster_keywords_list = cluster_keywords_dict.get(target_category_name, '').split(',')

            # append category_settings['use_main_keywords'] into the main_query list including the keywords from cluster_keywords_list
            if category_settings['use_main_keywords']:
                # Convert the string of keywords into a list
                main_keywords_list = category_settings['use_main_keywords'].split(",")  # assuming keywords are comma-separated
                cluster_keywords_list.extend(main_keywords_list)

            # Remove empty strings from the list and deduplicate
            cluster_keywords_list = list(set([keyword.strip() for keyword in cluster_keywords_list if keyword.strip()]))
            # Convert the list back to a comma-separated string and update the dictionary
            cluster_keywords_dict[target_category_name] = ','.join(cluster_keywords_list)
            cluster_keywords = cluster_keywords_dict

            # Creating the second string
            key_value_string = ', '.join([f"{key}, {value}" for key, value in cluster_keywords.items()])
            keywords_list = key_value_string.split(', ')
            unique_keywords_list = list(set(keywords_list))
            unique_keywords_string = ', '.join(unique_keywords_list)


            ### Save to the db the cluster keywords ###
            import query
            
            #Get keywords from the Processed_category table from column category_keywords and merge them with the new keywords
            category_keywords_records = db.session.query(Processed_category).filter_by(project_id=project_id, category_id=category_id).first()

            if category_keywords_records.category_keywords:
                category_keywords_list = category_keywords_records.category_keywords.split(', ')
                category_keywords_list = [keyword.lower().strip() for keyword in category_keywords_list]

                unique_keywords_list = unique_keywords_string.split(', ')
                unique_keywords_list = [keyword.lower().strip() for keyword in unique_keywords_list]

                distinct_keywords_set = set(category_keywords_list + unique_keywords_list)
                distinct_keywords_list = list(distinct_keywords_set)
                category_keywords = ', '.join(distinct_keywords_list)
                
                query.save_processed_category(db, Processed_category, project_id, category_id, category_keywords=category_keywords)
            else:
                query.save_processed_category(db, Processed_category, project_id, category_id, category_keywords=unique_keywords_string)



    ###############
    ### PROMPTS ###
    ###############

    class ContentSection:
        def __init__(self, condition, func, name, *args):
            self.condition = condition
            self.func = func
            self.name = name
            self.args = args
            self.content = ""
            self.token_count = 0

        def generate_content(self, system_prompt):
            if self.condition:
                # Emit the log for the execution of the section
                Config.socketio.emit('log', {'data': f'{formatted_now}: Writing: {self.name}'}, room=str(project_id), namespace='/')
                # Generate the prompt using the function associated with the section
                prompt = self.func(*self.args)
                if app_settings['print_prompt']:
                    Config.socketio.emit('log', {'data': f'{formatted_now}: System prompt:\n{system_prompt}'},room=str(project_id), namespace='/')
                    Config.socketio.emit('log', {'data': f'{formatted_now}: Prompt:\n{prompt}'},room=str(project_id), namespace='/')
                    return
                
                # Construct messages for the section using the system prompt and the generated prompt
                messages = construct_messages_for_section(system_prompt, prompt)
                # Generate the content for the section
                self.content, self.token_count = generate_part(db, Category_Settings, messages, app_settings, category_settings, seo_settings, results, project_id)


    # Define sections using the ContentSection class
    sections = [
        
        # Section for the intro prompt
        ContentSection(
            category_settings["include_intro"],
            compile_intro_prompt,
            "Introduction content",
            app_settings, category_settings, seo_settings, target_category_name, target_category_url, unique_keywords_string
        ),
        
        # Section for properties prompt, runs only if prop_output is present
        ContentSection(
            bool(prop_output),
            compile_properties_prompt,
            "Category Properties content",
            app_settings, seo_settings, target_category_name, prop_output, unique_keywords_string
        ),

        # Section for product levels prompt, runs only if "include_sales_info" is enabled in category_settings
        ContentSection(
            category_settings["include_sales_info"] and category_settings["add_best_selling_products"] > 0,
            compile_product_levels_intro_prompt,
            "Intro for best selling products content",
            category_settings, target_category_name, entry_level_products, mid_size_products, hi_end_products, app_settings, seo_settings, unique_keywords_string
        ),
    
        # Section for product levels prompt, runs only if "include_sales_info" is enabled in category_settings
        ContentSection(
            category_settings["include_sales_info"] and category_settings["add_best_selling_products"] > 0 and entry_level_products,
            compile_product_levels_entry_prompt,
            "Entry Levels content",
            category_settings, target_category_name, entry_level_products, mid_size_products, hi_end_products, app_settings, seo_settings, unique_keywords_string
        ),
        
        ContentSection(
            category_settings["include_sales_info"] and category_settings["add_best_selling_products"] > 0 and mid_size_products,
            compile_product_levels_mid_prompt,
            "Mid Levels content",
            category_settings, target_category_name, entry_level_products, mid_size_products, hi_end_products, app_settings, seo_settings, unique_keywords_string
        ),
        
        ContentSection(
            category_settings["include_sales_info"] and category_settings["add_best_selling_products"] > 0 and hi_end_products,
            compile_product_levels_high_prompt,
            "Hi-end Levels content",
            category_settings, target_category_name, entry_level_products, mid_size_products, hi_end_products, app_settings, seo_settings, unique_keywords_string
        ),
        
        # Section for top brands prompt, runs only if both "has_top_brands_info" and "add_top_brands" are true
        ContentSection(
            has_top_brands_info and category_settings["add_top_brands"] > 0,
            compile_top_brands,
            "Top Brands content",
            app_settings, seo_settings, target_category_name, top_brands, unique_keywords_string
        ),
        
        # Section for category levels prompt, runs only if "include_category_info" is enabled in category_settings
        ContentSection(
            category_settings["enable_additional_instructions"],
            compile_additional_info_prompt,
            "Additional info content",
            app_settings, category_settings, seo_settings, unique_keywords_string, target_category_name
        ),
        
        # Section for category levels prompt, runs only if "include_category_info" is enabled in category_settings
        ContentSection(
            category_settings["include_category_info"],
            compile_category_levels,
            "Category Levels content",
            app_settings, seo_settings, top_level_category, same_level_categories, sub_level_categories, unique_keywords_string, target_category_name
        )
    ]


    system_prompt = (
        'The output must be a coherent and detailed description of the category in question in strictly and valid HTML format. It should '
        f'be written in {app_settings["language"]}. The output should contains valid HTML code except tags like H1, newline, body and other main tags.'
        'For the headings use H3 and before each heading add one additional new line for better readability.'
        'For bold use strong tag, for italic use em tag.'
        f'The output must be ONLY the description explicitly and you should keep the {app_settings["language"]} language from the prompt!'
    )

    # Generate content for each section
    for section in sections:
        section.generate_content(system_prompt)

    # Calculate the total tokens used across all sections
    
    # Query the Processed_category table for the column token_count for this category
    record = db.session.query(Processed_category).filter_by(project_id=project_id, category_id=category_id).first()
    current_token_count = record.token_count if record else 0

    if current_token_count is None or current_token_count == 0:
        total_tokens_used = sum([section.token_count for section in sections])
        import query
        query.save_processed_category(db, Processed_category, project_id, category_id, token_count=total_tokens_used)
    else:
        total_tokens_used = sum([section.token_count for section in sections])
        updated_token_count = current_token_count + total_tokens_used
        import query
        query.save_processed_category(db, Processed_category, project_id, category_id, token_count=updated_token_count)

    # Combine the content of all sections
    final_article = "".join([section.content for section in sections])

    return final_article


def generate_part(db, Category_Settings, messages, app_settings, category_settings, seo_settings, results, project_id):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
    category_id = category_settings['category_id']
    for attempt in range(15):
        ### STOP PROCESS IF USER CLICKS STOP ###
        if stop_category_process.get(project_id, False):
            stop_category(project_id)  # Stop process
            Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'},room=str(project_id), namespace='/')
            project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
            if project:
                project.in_progress = False
                db.session.commit()
            return 
        try:
            
            response = openai.ChatCompletion.create(
                model=app_settings['model'],
                messages=messages,
                temperature=0,
            )
            
            # If the request was successful, break out of the loop
            break
        except openai.error.AuthenticationError:
            # Handle authentication errors (e.g., invalid API key)
            Config.socketio.emit('log', {'data': 'Authentication Error. Check your OpenAI API Key!'}, room=str(1), namespace='/')
            break
        except (openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError, ) as e:  
            # Handle APIError, Timeout, and ServiceUnavailableError for retry
            wait_time = 2 * (attempt + 1)
            Config.socketio.emit('log', {'data': f'{formatted_now}: Encountered an issue: {e.error}. Waiting for {wait_time} seconds before retrying.'},room=str(project_id), namespace='/')
            time.sleep(wait_time)
        except (openai.error.InvalidRequestError) as e:
            Config.socketio.emit('log', {'data': f'{formatted_now}: {e}'},room=str(project_id), namespace='/')
            break
        except Exception as e:
            # Handle all other exceptions without retrying
            Config.socketio.emit('log', {'data': f'{formatted_now}: {e}'},room=str(project_id), namespace='/')

            break
    else:
        raise Exception("Maximum number of retries exceeded.")

    # Get the main content
    total_tokens = response['usage']['total_tokens']
    main_content = response['choices'][0]['message']['content']

    return main_content, total_tokens

def processCategories(db, Processed_category, Category_Settings, project_id, x_cloudcart_apikey, store_url, model):
    # Fetch all processed category IDs for the given project
    processed_categories = db.session.query(Processed_category.category_id).filter_by(project_id=project_id).all()
    processed_category_ids = {category[0] for category in processed_categories}

    # Fetch all categories
    app_settings = {
        'X-CloudCart-ApiKey': x_cloudcart_apikey,
        'url': store_url,
    }
    categories = getCategoryDetails(app_settings, fields=['id', 'name', 'url_handle'])

    # Filter out categories that have already been processed
    unprocessed_categories = [category for category in categories if int(category.get('id')) not in processed_category_ids]


    # Iterate and save each unprocessed category
    for category in unprocessed_categories:
        category_id = category.get('id', None)

        #if not has_products_in_category(app_settings, category_id):
        #    print(f"Skipping category ID {category_id} because it has no products.")
        #    continue  # skip to the next category if no products

        category_name = category.get('name', None)
        category_url_handle = category.get('url_handle', None)
        if category_url_handle:
            full_category_url = f"{store_url}/category/{category_url_handle}"
        else:
            full_category_url = None
        
        # Since the description is not fetched in the current API call, it will be None.
        # If you want to fetch it, adjust the fields in the getCategoryDetails call above.
        description = category.get('description', None)

        new_category = Processed_category(
            project_id=project_id,  
            category_id=category_id, 
            category_name=category_name.lower(), 
            category_url=full_category_url,
            category_description=description
        )
        
        db.session.add(new_category)

    db.session.commit()





def cat(db, Processed_category, Category_Settings, app_settings, category_settings, seo_settings, project_id):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
    category_id = category_settings['category_id']
    Config.socketio.emit('log', {'data': f'{formatted_now}: Started...'},room=str(project_id), namespace='/')
    project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
    if project:
        project.in_progress = True
        db.session.commit()

    stop_category_process[project_id] = False
    # Get category IDs
    if not category_settings['category_id']:
        Config.socketio.emit('log', {'data': f'{formatted_now}: No specific category ID provided. The script will run for all categories.'}, room=str(project_id), namespace='/')
        category_ids = getCategoryIds(app_settings)

        # Ensure category_ids is sorted in ascending order
        category_ids.sort()

        # Get the last processed category ID
        last_processed_id = get_last_processed_category(db, Processed_category, project_id)
        
        if last_processed_id:
            # Convert the last_processed_id to a string
            str_last_processed_id = str(last_processed_id)

            # Find the index of the last processed category ID in the category_ids list
            try:
                index_of_last_processed = category_ids.index(str_last_processed_id)
                
                # Slice the list to start from the next category ID
                category_ids = category_ids[index_of_last_processed + 1:]
                
            except ValueError:
                # This means the last processed category ID was not found in the category_ids list
                # So, continue processing all category IDs
                pass


    else:
        category_ids = [category_settings['category_id']]

    # Prepare a list to hold all results
    results = []

    # Iterate over category IDs
    for category_id in category_ids:
        Config.socketio.emit('log', {'data': f'{formatted_now}: Processing category ID: {category_id} '},room=str(project_id), namespace='/')
        project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
        if project:
            project.in_progress = True
            db.session.commit()

        ### STOP PROCESS IF USER CLICKS STOP ###
        if stop_category_process.get(project_id, False):
            stop_category(project_id)  # Stop process
            Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'},room=str(project_id), namespace='/')
            project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
            if project:
                project.in_progress = False
                db.session.commit()
            return 
        
        # Get category info
        target_category_info = getTargetCategoryInfo(db, Processed_category, app_settings, category_id, category_settings, seo_settings, project_id)

        ### STOP PROCESS IF USER CLICKS STOP ###
        if stop_category_process.get(project_id, False):
            stop_category(project_id)  # Stop process
            Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'}, room=str(project_id), namespace='/')
            project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
            if project:
                project.in_progress = False
                db.session.commit()
            return

        categories_by_id, children_by_parent_id = getCategoryInfo(app_settings)

        result = {}
        result['target_category_info'] = target_category_info  # 'name', 'description', 'url_handle'

        if category_settings['include_category_info'] or category_settings['include_category_info_faq']:
            category_info = getCategoryLevels(app_settings, category_id, categories_by_id, children_by_parent_id)
            result['category_info'] = category_info


        if category_settings['include_sales_info'] or category_settings['include_faq_info']:
            # Check if both add_best_selling_products and add_top_brands are not 0
            if category_settings['add_best_selling_products'] != 0 or category_settings['add_top_brands'] != 0 or category_settings['add_best_selling_products_faq'] or category_settings['add_top_brands_faq']:

                products_by_sales, best_selling_brands = getOrderedProductsbySales(category_id, app_settings, category_settings, target_category_info)

                result['products_by_sales'] = products_by_sales
                result['best_selling_brands'] = best_selling_brands
        


        # Save the processed category
        # This is information about all fields from the database
        # project_id, category_id, category_structure, category_name, category_prompt, category_description, category_faqs, category_keywords, category_custom_keywords
        query.save_processed_category(db, Processed_category, project_id, category_id, category_structure=result)     

        ### STOP PROCESS IF USER CLICKS STOP ###
        if stop_category_process.get(project_id, False):
            stop_category(project_id)  # Stop process
            Config.socketio.emit('log', {'data': f'{formatted_now}: Process stopped by user.'},room=str(project_id), namespace='/')
            project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
            if project:
                project.in_progress = False
                db.session.commit()
            return         
        # Add the result for the current category to the list of results
        results.append(result)
        description = ''
        if app_settings['enable_category_description']:
            description = generate_category_description(db, Category_Settings, Processed_category, app_settings, category_settings, seo_settings, result, project_id)
            # Save the processed category
            # This is information about all fields from the database
            # project_id, category_id, category_structure, category_prompt, category_description, category_faqs, category_keywords, category_custom_keywords
            if app_settings['test_mode']:
                query.save_processed_category(db, Processed_category, project_id, category_id, category_description=description, category_test_mode=True)   
            else:
                query.save_processed_category(db, Processed_category, project_id, category_id, category_description=description, category_test_mode=False) 
        if category_settings['enable_faq_generation']:
            faq = craft_faqs(db, Processed_category, Category_Settings, app_settings, category_settings, seo_settings, result, project_id)
            # Save the processed category
            # This is information about all fields from the database
            # project_id, category_id, category_structure, category_prompt, category_description, category_faqs, category_keywords, category_custom_keywords
            if app_settings['test_mode']:
                query.save_processed_category(db, Processed_category, project_id, category_id, category_faqs=faq, category_test_mode=True)
            else:
                query.save_processed_category(db, Processed_category, project_id, category_id, category_faqs=faq, category_test_mode=False)   
        
        ### Combine description with FAQ into one variable DESCRIPTION ###
        description = description or ""
        if app_settings['print_prompt'] == False:
            if app_settings['enable_category_description'] and category_settings['enable_faq_generation']:
                description = description + '\n' + faq
            elif category_settings['enable_faq_generation'] and category_settings['append_faq']:
                get_category_description = getTargetCategoryInfo(db, Processed_category, app_settings, category_id, category_settings, seo_settings, project_id, include_description=True)
                # set the description into a string
                description = get_category_description['description'][0] + '\n\n' + faq
            elif category_settings['enable_faq_generation']:
                description = faq

        # Update the category description
        if app_settings['test_mode']:
            if app_settings['print_prompt'] == False:
                Config.socketio.emit('log', {'data': f'{formatted_now}: Result completed: \n{description} '},room=str(project_id), namespace='/')
                project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
                if project:
                    project.in_progress = False
                    db.session.commit()
            else:
                Config.socketio.emit('log', {'data': f'{formatted_now}: Result completed: \n{description} '},room=str(project_id), namespace='/')
                project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
                if project:
                    project.in_progress = False
                    db.session.commit()
        else:
            if app_settings['print_prompt'] == True:
                Config.socketio.emit('log', {'data': f'{formatted_now}: Result completed: \n{description} '},room=str(project_id), namespace='/')
                project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
                if project:
                    project.in_progress = False
                    db.session.commit()
            else:
                Config.socketio.emit('log', {'data': f'{formatted_now}: Updating category description for category ID: {category_id} '},room=str(project_id), namespace='/')
                updateCategory(category_id, description, app_settings, project_id)
                Config.socketio.emit('log', {'data': f'{formatted_now}: Proces completed. Updated category ID: {category_id} '},room=str(project_id), namespace='/')
                # Return the list of results after the loop has finished
                project = Category_Settings.query.filter_by(project_id=project_id, category_id=category_id).first()
                if project:
                    project.in_progress = False
                    project.category_ready = False
                    db.session.commit()
                category_updated = Processed_category.query.filter_by(project_id=project_id, category_id=category_id).first()
                if category_updated:
                    category_updated.category_update = True
                    db.session.commit()
    return

def updateCategory(category_id, description, app_settings, project_id):
    now = datetime.now()
    formatted_now = now.strftime("%d/%m/%Y %H:%M:%S")
    headers = {
        'X-CloudCart-ApiKey': app_settings['X-CloudCart-ApiKey'],
        'Content-Type': 'application/vnd.api+json',
    }
    url = f"{app_settings['url']}/api/v2/categories/{category_id}"

    attributes = {
        "description": description
    }   

    body = {
        "data": {
            "type": "categories",
            "id": str(category_id),
            "attributes": attributes
        }
    }

    max_retries = 15

    for attempt in range(max_retries):
        try:
            response = requests.patch(url, data=json.dumps(body), headers=headers)
            if response.status_code in (429, 500, 502, 503):  # Retry for status codes 500 and 503
                raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")
            elif response.status_code != 200:  # For other non-200 status codes, fail immediately
                raise Exception(f"Request to {url} failed with status code {response.status_code}. The response was: {response.text}")
            
            return response.json()  # If request was successful, break out of the loop and return the response
        except Exception as e:
            if attempt < max_retries - 1:  # If it's not the last attempt, wait and then continue to the next iteration
                wait_time = 5 * (attempt + 1)
                Config.socketio.emit('log', {'data': f"{formatted_now}: Error occured at CloudCart. Waiting for {wait_time} seconds before retrying."},room=str(project_id), namespace='/')
                time.sleep(wait_time)
            else:  # On the last attempt, fail with an exception
                raise
   
def stop_category(project_id):
    print(f'Stopping the process for project {project_id}...')
    stop_category_process[project_id] = True  

#main_query = "ЕСПРЕСО КАФЕМАШИНИ, Чаши, Шишета"
#project_id = 2
#result = keyword_clustersTEST(db, Category_Settings, app_settings, main_query, project_id)
#result = keyword_clusters(app_settings, main_query)
#print(result)

import json
import os
import requests
from googlesearch import search
from bs4 import BeautifulSoup
from PIL import Image
import io
import base64
from urllib.parse import urlparse, urlunparse
from openai import OpenAI
import geopandas as gpd
from shapely.geometry import Point
from sweref99 import projections
from requests.exceptions import Timeout
from urllib.parse import unquote
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import random
from json.decoder import JSONDecodeError


# Load the Swedish county shapefile data
counties = gpd.read_file("counties/Lan_Sweref99TM_region.shp")

def remove_duplicate_words_in_string(string):
    iterable = string.split()
    seen = set()
    unique_words = []
    
    for word in iterable:
        # Check if the word is not in the set (not seen before)
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    # Join the unique words to form the processed string
    processed_string = ' '.join(unique_words)
    
    return processed_string

def process_url(url):
   # Remove domain name endings (e.g., .com, .org)
    url = re.sub(r'\.(?![^/]*\.)[^/]+', '', url)
    print(url)
    # Remove domain name endings (e.g., .com, .org)
    url = re.sub(r'\.[a-zA-Z]+$', '', url)
    
    # Remove "https" or "http"
    url = re.sub(r'^https?://', '', url, flags=re.IGNORECASE)

    # Remove "www"
    url = re.sub('www', '', url, flags=re.IGNORECASE)

    # Replace dots with blank spaces
    url = url.replace('.', ' ')

    # Replace - with blank spaces
    url = url.replace('-', ' ')
    # Replace - with blank spaces
    url = url.replace('/', ' ')

    # Remove special characters
    url = re.sub(r'[^a-zA-Z0-9\s]', '', url)

    url = remove_duplicate_words_in_string(url)
    return url

# Define a function to find the county for a given latitude and longitude
def find_county(lat, lon):
    # Convert the WGS 84 coordinates to SWEREF 99 using the latlon_to_rt90 function
    tm = projections.make_transverse_mercator("SWEREF_99_TM")
    northing, easting = tm.geodetic_to_grid(lat, lon)

    # Create a shapely Point object from the transformed coordinates
    point = Point(easting, northing)

    # Loop through the counties and check if the point is inside each polygon
    for i in range(len(counties)):
        if point.within(counties.iloc[i].geometry):
            return counties.iloc[i].LnNamn

    # If the point is not inside any polygon, return None
    return None

def import_json(json_file):
  with open(json_file, "r", encoding="utf-8") as f:
    return json.load(f)

def import_not_transformed(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter the data to include only records where is_transformed is either False or not present
    filtered_data = [entry for entry in data if 'is_transformed' not in entry or entry['is_transformed'] is not True]

    return filtered_data

def import_not_approved(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter the data to include only records where is_approved is either False or not present
    filtered_data = [entry for entry in data if 'is_approved' not in entry or entry['is_approved'] is not True]

    return filtered_data

def get_lat_long_goog(api_key, *search_strings):
    for search_string in search_strings:
        print(f"Trying to geocode '{search_string}'")
        url = f'https://maps.googleapis.com/maps/api/geocode/json?address={search_string}&key={api_key}'
        try:
            response = requests.get(url)
            coords = response.json()

            # Get the geocoordinates from the results
            latitude = coords['results'][0]['geometry']['location']['lat']
            longitude = coords['results'][0]['geometry']['location']['lng']

            return [latitude, longitude]

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"Received 429 error. Waiting before retrying...")
                time.sleep(60)  # Wait for 1 minute (adjust as needed)
            else:
                print(f"An HTTP error occurred during geocoding for '{search_string}': {e}")
        except Exception as e:
            if e == "list index out of range":
                print(f"No results found for '{search_string}'")
            else:
                print(f"An error occurred during geocoding for '{search_string}': {e}")

    # Return [0, 0] if no coordinates were found
    return [0, 0]

def check_allowed_url_get_goog(disallowed_urls, query):

    try:
        # Using the googlesearch library to perform a search and get the results
        results_generator = search(query, num_results=5)
        # Convert the generator to a list
        urls = list(results_generator)

        print(urls)
        print(f"Found {len(urls)} search results.")

        # Use the list for further processing if needed
        for url in urls:
            print(url)
        # Filter out URLs that start with specified prefixes
        filtered_urls = [url for url in urls if not url.startswith(tuple(disallowed_urls))]
        print(filtered_urls)
        if filtered_urls:
            # Return the first URL that doesn't start with specified prefixes
            return filtered_urls[0]
        else:
            print("No suitable URL found.")
            return None

    except requests.exceptions.HTTPError as e:
        if "429" in str(e):
            print(f"Received 429 error. Waiting before retrying...")
        else:
            print(f"An HTTP error occurred while searching: {e}")
    except Exception as e:
        print(f"An error occurred while searching: {e}")

    print("No suitable URL found.")
    return None

def check_allowed_url_get_bing(disallowed_urls, query):

    try:
        # Using the get_bing_search_results function to perform a Bing search and get the results
        results = get_bing_search_results(query)
        print(f"Found {len(results)} search results.")

        # Filter out URLs that start with specified prefixes
        filtered_urls = [result for result in results if not result.startswith(tuple(disallowed_urls))]

        if filtered_urls:
            # Return the first URL that doesn't start with specified prefixes
            return filtered_urls[0]
        else:
            print("No suitable URL found.")
            return None

    except requests.exceptions.HTTPError as e:
        if "429" in str(e):
            print(f"Received 429 error. Waiting before retrying...")
        else:
            print(f"An HTTP error occurred while searching: {e}")
    except Exception as e:
        print(f"An error occurred while searching: {e}")

    print("No suitable URL found.")
    return None

def check_allowed_url(url, query):
    disallowed_urls = {
        "https://www.lopplistan.se",
        "https://www.trailrunningsweden.se",
        "https://www.löpning.se",
        "https://www.loppkartan.se",
        "https://www.jogg.se",
        "https://raceid.com",
        "https://www.raceone.com",
    }

    if not url.startswith(tuple(disallowed_urls)) and url != "":
        print(f"The incoming URL '{url}' is not disallowed or empty. Returning it directly.")
        return url

    number_of_retries = 2  # Number of retries
    for _ in range(number_of_retries):
        # Try Google search first
        google_result = check_allowed_url_get_goog(disallowed_urls, query)
        if google_result:
            return google_result

        # If Google search fails, try Bing search
        bing_result = check_allowed_url_get_bing(disallowed_urls, query)
        if bing_result:
            return bing_result

        # If both searches fail, pause for 30 seconds and try Google search once more
        if _ <= number_of_retries - 1:
            print("Both Google and Bing searches failed. Waiting for 30 seconds before retrying...")
            time.sleep(30)

    return None

# Function to get Bing search results
def get_bing_search_results(query):
    try:
        # Replace spaces with hyphens in the query
        query = query.replace(" ", "%20")

        # Bing search URL
        bing_url = f"https://www.bing.com/search?q={query}"

        # Send a GET request to Bing
        response = requests.get(bing_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the element with id="b_results"
        rso_element = soup.find(id='b_results')

        # Check if the 'b_results' element exists
        if rso_element:
            # Find the first 5 child list items of 'b_results'
            child_items = rso_element.find_all('li', recursive=False)[:5]

            # Extract the first href from each child list item
            hrefs = [li.find('a')['href'] for li in child_items if li.find('a')]

            return hrefs

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")

    return None

#def check_allowed_url_get_goog_selenium(url, query):
    # Check if the incoming URL is not part of the specified set
    disallowed_urls = {
        "https://www.lopplistan.se",
        "https://www.trailrunningsweden.se",
        "https://www.löpning.se",
        "https://www.loppkartan.se",
        "https://www.jogg.se",
        "https://raceid.com",
    }

    if not url.startswith(tuple(disallowed_urls)) and url != "":
        print(f"The incoming URL '{url}' is not disallowed or empty. Returning it directly.")
        return url

    try:
        retry_count = 3  # Number of retries
        for _ in range(retry_count):
            print(f"Trying to search for {query}")
            try:
                # Set up Selenium WebDriver
                options = Options()
                #options.add_argument("--headless=new")
                options.add_argument("user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'")
                driver=webdriver.Chrome(options=options)

                # Navigate to Google and perform a search
                driver.get("https://www.google.com")
                print("1")

                # Switch to the popup window
                popup_window_handle = None

                # Wait for the popup to appear and get its window handle
                try:
                    popup_window_handle = WebDriverWait(driver, 10).until(
                        EC.number_of_windows_to_be(2)
                    )
                except TimeoutException:
                    print("Popup window did not appear within 10 seconds")

                if popup_window_handle:
                    # Switch to the popup window
                    driver.switch_to.window(popup_window_handle[1])

                    try:
                        # Wait for the "Godkänn alla" button to appear in the popup
                        accept_all_button = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Godkänn alla')]"))
                        )

                        # Click the "Godkänn alla" button
                        accept_all_button.click()
                        print("Clicked 'Godkänn alla' in the popup")
                    except TimeoutException:
                        print("'Godkänn alla' button not found within 5 seconds in the popup")

                    # Switch back to the main window
                    driver.switch_to.window(popup_window_handle[0])

                # Wait for the search box to be present and clickable
                search_box = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.NAME, "q"))
                )
                time.sleep(100)
                print("1")
                search_box.send_keys(query)
                print("1")
                search_box.send_keys(Keys.RETURN)
                print("1")

                # Wait for some time to let the page load
                time.sleep(random.uniform(3, 5))  # Adjust sleep time as needed

                # Extract URLs from search results
                search_results = driver.find_elements_by_css_selector("div#search a")
                print(f"Found {len(search_results)} search results.")
                urls = [result.get_attribute("href") for result in search_results]

                # Filter out URLs that start with specified prefixes
                filtered_urls = [url for url in urls if not url.startswith(tuple(disallowed_urls))]

                if filtered_urls:
                    # Return the first URL that doesn't start with specified prefixes
                    return filtered_urls[0]
                else:
                    print("No suitable URL found.")
                    return None

            except Exception as e:
                print(f"An error occurred while searching: {e}")
            finally:
                # Close the WebDriver
                driver.quit()
                
            # Retry after waiting for a random time
            wait_time = random.uniform(60, 120)  # Adjust wait time as needed
            print(f"Waiting for {wait_time} seconds before retrying...")
            time.sleep(wait_time)

        print("No suitable URL found after retries.")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def add_https_if_missing(url):
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        # If the scheme (http/https) is missing, add https://
        parsed_url = parsed_url._replace(scheme="https")
    return urlunparse(parsed_url)

def convert_and_compress_image(image_url, max_size_kb=200):
    try:
        # Add https:// if the scheme is missing
        image_url = add_https_if_missing(image_url)
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()

        # Open the image using PIL
        img = Image.open(io.BytesIO(response.content))

        # Convert the image to WebP format
        webp_image = img.convert("RGB")

        # Compress the image to the specified maximum size (in bytes)
        target_size_bytes = max_size_kb * 1024
        webp_data = io.BytesIO()
        initial_quality = 85
        webp_image.save(webp_data, "WEBP", quality=initial_quality)  # You can adjust the quality parameter
        while webp_data.tell() > target_size_bytes:
            print(f"Compressed image size: {webp_data.tell() / 1024} KB")
            webp_data = io.BytesIO()
            initial_quality -= 15
            webp_image.save(webp_data, "WEBP", quality=initial_quality)
        # Encode the compressed image data in base64
        base64_data = base64.b64encode(webp_data.getvalue()).decode('utf-8')
        
        return base64_data

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return None

def get_website_contents(url):
    #Possibly extend this to also crawl images from google
    website_contents = {
        "title": "",
        "description": "",
        "h1": "",
        "p": [],
        "h2": [],
        "images": [],
        "url": url
    }

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        website_contents["title"] = soup.title.text.strip() if soup.title else ""
        website_contents["description"] = soup.find('meta', {'name': 'description'})['content'].strip() if soup.find('meta', {'name': 'description'}) else ""
        website_contents["h1"] = soup.find('h1').text.strip() if soup.find('h1') else ""
        
        # Extract up to 5 paragraphs
        paragraphs = soup.find_all('p')
        website_contents["p"] = [p.text.strip() for p in paragraphs[:5]]

        # Extract up to 5 h2 headings
        h2_headings = soup.find_all('h2')
        website_contents["h2"] = [h2.text.strip() for h2 in h2_headings[:5]]

        # Extract up to 4 images, convert to WebP, and compress
       # image_tags = soup.find_all('img')
        #print(url)
        #for img_tag in image_tags:
        #    # Check if 'src' attribute exists
        #    if 'src' in img_tag.attrs:
        #        image_url = img_tag['src']
        #        compressed_image_data = convert_and_compress_image(image_url)
        #        if compressed_image_data:
        #            website_contents["images"].append(compressed_image_data)
#
        #        # Break the loop when you have collected 4 compressed images
        #        if len(website_contents["images"]) >= 4:
        #            break

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")

    except Exception as e:
        print(f"An error occurred while parsing the HTML: {e}")

    return website_contents

### OPENAI ###
token_price_input = 0.001
token_price_output = 0.002

def get_completion(prompt, openai_key, costometer, timeout=20):
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(api_key=openai_key)

    try:
        # Non-streaming:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            timeout=25  # Set the timeout to 25 seonds
        )
        
        return completion.choices[0].message.content, costometer + token_price_input * len(prompt) + token_price_output * len(completion.choices[0].message.content)

    except Timeout:
        # Handle timeout, return None or raise an exception as needed
        print("Request timed out")
        return None, costometer  # You may need to adjust the return values as needed

def get_images_selenium(search_term):
    # Set up the Selenium WebDriver (make sure you have the appropriate driver installed)
    #setup
    print("getting images")
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("user-agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'")
    driver=webdriver.Chrome(options=options)
    driver.get("https://www.bing.com/images")

    # Find the search input field and send the search term
    search_box = driver.find_element("name", "q")
    search_box.send_keys(search_term)
    search_box.send_keys(Keys.RETURN)

    # Wait for some time to let the page load
    time.sleep(2)
    print(driver.current_url)

    # Extract the HTML content after the search
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract the first 4 image thumbnails
    thumbnails = []
    for raw_img in soup.find_all('a', class_='iusc'):
        link = raw_img.get('href')
        print(link[0:105])

        if link and link.find("&mediaurl="):
            # Find the start and end index of the substring containing img path
            start_index = link.find("&mediaurl=") + len("&mediaurl=")
            end_index = link.find("&cdnurl")
            img_url = link[start_index:end_index]

            # Decode the URL string
            decoded_img_url = unquote(img_url)
            thumbnails.append(decoded_img_url)
        if len(thumbnails) == 12:
            break

    # Close the WebDriver
    driver.quit()

    # Only return 1 to 4 as the first seems weird
    return thumbnails

def return_if_exists(in_dict,file_path, id):
    with open(file_path, encoding='utf-8') as f:
        source = json.load(f)
    for j in range(len(source)):
        if in_dict[id] == source[j][id]:
            return source[j]
    return None

def map_distance(distance, race_type):
    if race_type in ['backyard', 'relay']:
        return race_type  # Capitalize the type for backyard and relay
    if race_type == 'track':
        return f"{distance} meter"  # Concatenate distance and "meter" for track races
    if race_type == 'road' and type(distance) != str:
        if 21000 <= distance <= 21500:
            return 'Halvmarathon'
        elif 42000 <= distance <= 42500:
            return 'Marathon'
        elif 80000 <= distance <= 82000:
            return '50 miles'
        elif 160000 <= distance <= 170000:
            return '100 miles'
        elif 300000 <= distance <= 340000:
            return '200 miles'
        elif distance > 42000:
            return 'Ultramarathon'
        else: 
            return f"{round(distance / 1000)} km"
    elif race_type in ['trail', 'terrain'] and type(distance) != str:
        if 21000 <= distance <= 21500:
            return 'Trail Halvmarathon'
        elif 42000 <= distance <= 42500:
            return 'Trail Marathon'
        elif 80000 <= distance <= 82000:
            return '50 miles'
        elif 160000 <= distance <= 170000:
            return '100 miles'
        elif 300000 <= distance <= 340000:
            return '200 miles'
        elif distance > 42000:
            return 'Trail Ultramarathon'
        else: 
            return f"{round(distance / 1000)} km"
    # If none of the above conditions match, return 'Unknown'
    return 'Unknown'

def race_category_mapping(distances, race_type):
    race_categories = []
    for distance in distances:
        race_categories.append(map_distance(distance, race_type))
    
    return ', '.join(race_categories)

def get_all_ids_from_json(file_path):
    try:
        # If yes, load the existing data
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        # Assuming the JSON data is a list of dictionaries
        id_values = [item.get('id') for item in data]

        return id_values
    except:
        # Handle the case where the file is empty or not valid JSON
        print(f"Error decoding JSON in {file_path}. Returning empty list.")
        return []


def add_or_update_object(name, obj_id, images_dict, json_file_path, override_current=False):
    """
    Add or update an object in a JSON file.

    Parameters:
        - name: The name to add/update.
        - obj_id: The id to add/update.
        - images_dict: The images dictionary to add/update.
        - json_file_path: The path to the JSON file.
        - override_current: If True, override the existing object with the same id.

    Returns:
        None
    """
    # Check if the JSON file exists
    if not os.path.exists(json_file_path):
        # If not, create an empty list as the initial data
        data = []
    else:
        # If yes, load the existing data
        with open(json_file_path, encoding='utf-8') as f:
            data = json.load(f)

    # Check if there is an object with the same id
    existing_object_index = None
    for index, item in enumerate(data):
        if isinstance(item, dict) and item.get('id') == obj_id:
            existing_object_index = index
            break

    if existing_object_index is not None and override_current:
        # Append name, id, and images dict to the existing object
        existing_object = data[existing_object_index]
        existing_object['name'] = name
        existing_object['id'] = obj_id
        existing_object['images'] = images_dict
    else:
        # Create a new object with name, id, and images dict
        new_object = {'name': name, 'id': obj_id, 'images': images_dict}
        data.append(new_object)

    # Write the modified data back to the JSON file
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

if __name__ == "__main__":
    
    test = get_all_ids_from_json("staged_for_approval.json")
    print(test)
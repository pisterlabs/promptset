import sys
sys.path.append('/Users/pranav/anaconda3/lib/python3.11/site-packages')
sys.path.append('//Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages')

import re
import ssl
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import tldextract
import whois
from datetime import datetime
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from apikey import APIKEY2
import os
from Seo import seoStats
from Seo2 import indexAndStats

def extract_features(url):
    # Perform a WHOIS lookup for the domain
    res = whois.whois(url)

    # Get the html contents of the webpage
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
    
    # Initialize a dictionary to store feature values
    features = {}

    # Feature 1: Check if the URL contains an IP address
    if re.match(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", url):
        features['having_IPhaving_IP_Address'] = -1
    else:
        features['having_IPhaving_IP_Address'] = 1

    # Feature 2: Check URL length
    if len(url) < 54:
        features['URLURL_Length'] = 1
    elif 54 <= len(url) <= 75:
        features['URLURL_Length'] = 0
    else:
        features['URLURL_Length'] = -1

    # Feature 3: Check for URL shortening service (e.g., TinyURL)
    shortening_services = ["tinyurl.com", "bit.ly", "ow.ly", "t.co", "is.gd", "shorte.st", "goo.gl", "buff.ly", "rebrand.ly"]
    for service in shortening_services:
        if service in url:
            features['Shortining_Service'] = -1
        else:
            features['Shortining_Service'] = 1

    # Feature 4: Check for '@' symbol in the URL
    if "@" in url:
        features['having_At_Symbol'] = -1
    else:
        features['having_At_Symbol'] = 1

    # Feature 5: Check for double slash redirection
    if "//" in urlparse(url).path:
        # Check the position of the last occurrence of "//" in the URL
        last_slash_position = url.rfind("//")
        if url.startswith("http://") and last_slash_position > 6:
            features['double_slash_redirecting'] = -1
        elif url.startswith("https://") and last_slash_position > 7:
            features['double_slash_redirecting'] = -1
        else:
            features['double_slash_redirecting'] = 1
    else:
        features['double_slash_redirecting'] = 1

    # Feature 6: Check for prefix-suffix separation by '-' in the domain
    if '-' in urlparse(url).netloc:
        features['Prefix_Suffix'] = -1
    else:
        features['Prefix_Suffix'] = 1

    # Feature 7: Count subdomains
    subdomains = urlparse(url).netloc.split('.')
    if len(subdomains) == 2:
        features['having_Sub_Domain'] = 1
    elif len(subdomains) == 3:
        features['having_Sub_Domain'] = 0
    else:
        features['having_Sub_Domain'] = -1

    # Feature 8: To check SSL certificate
    having_ssl = None
    try:
        # Parse the URL to get the hostname
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname

        # Get the SSL certificate
        cert = ssl.get_server_certificate((hostname, 443))

        # If no exceptions were raised, the URL has an SSL certificate
        having_ssl = True
    except Exception as e:
        # An exception was raised, indicating no SSL certificate
        having_ssl = False
    
    if having_ssl == True:
        features['SSLfinal_State'] = 1
    else:
        features['SSLfinal_State'] = -1

    # Feature 16: Check for domain expiration
    # Extract the expiration date from the WHOIS response
    if "expiration_date" in res:
            expiration_date = res["expiration_date"]
            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]  # Replace with the actual expiration date from WHOIS

    # Parse the expiration date string into a datetime object
    expiration_date = datetime.strptime(str(expiration_date), "%Y-%m-%d %H:%M:%S")

    # Calculate the time remaining until expiration
    current_date = datetime.now()
    time_until_expiration = expiration_date - current_date

    # Convert the time remaining to years
    years_until_expiration = time_until_expiration.days / 365

    # If the domain expires in less than or equal to 1 year, mark it as phishing
    if years_until_expiration <= 1:
        features['Domain_registeration_length'] = -1
    else:
        features['Domain_registeration_length'] = 1

    # Feature 8: Check for HTTPS in the URL
    if url.startswith("https://"):
        features['HTTPS_token'] = 1
    else:
        features['HTTPS_token'] = -1

    # Feature 9: Check for request URLS
    extracted = tldextract.extract(url)
    main_domain = f"{extracted.domain}"

    # Get the html contents of the webpage
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

    external_urls = 0  # Counter for external URLs
    total_urls = 0     # Counter for total URLs

    for link in soup.find_all('a', href=True):
        href = link['href']
        total_urls += 1

        # Check if the URL is external (different domain)
        parsed_url = urlparse(href)
        if main_domain not in parsed_url.netloc:
            external_urls += 1

    # Calculate the percentage of external URLs

    percentage_external_urls = 0

    try:
        percentage_external_urls = (external_urls / total_urls) * 100
    except Exception as e:
        features['Request_url'] = -1
        features['URL_of_Anchor'] = -1
        features['Links_in_tags'] = -1

    # Determine the category based on the percentage
    if percentage_external_urls < 40:
        features['Request_url'] = 1
        features['URL_of_Anchor'] = 1
        features['Links_in_tags'] = 1
    elif 40 <= percentage_external_urls < 60:
        features['Request_url'] = 0
        features['URL_of_Anchor'] = 0
        features['Links_in_tags'] = 0
    else:
        features['Request_url'] = -1
        features['URL_of_Anchor'] = -1
        features['Links_in_tags'] = -1

    # Feature 9: Check SFH
    extracted = tldextract.extract(url)
    main_domain = f"{extracted.domain}"
    try:
        # Get the html contents of the webpage
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all form elements in the HTML
            form_elements = soup.find_all('form')

            # Initialize SFH classification
            sfh_classification = 1  # Default: Legitimate

            # Check each form element for the action attribute (SFH)
            for form in form_elements:
                sfh = form.get('action', '')

                # Check if the SFH is empty or "about:blank"
                if not sfh.strip() or sfh.strip().lower() == 'about:blank':
                    sfh_classification = -1  # Phishing

            # Add SFH classification to the features dictionary with labels
            if sfh_classification == 1:
                features['SFH'] = 1  # Legitimate
            elif sfh_classification == 0:
                features['SFH'] = 0  # Suspicious
            else:
                features['SFH'] = -1  # Phishing

        else:
            print(f"Failed to retrieve the website. Status code: {response.status_code}")
            features['SFH'] = -1  # Phishing

    except Exception as e:
        print(f"An error occurred: {e}")
        features['SFH'] = -1  # Phishing

    # Feature 10: Submitting Information to Email
    try:
        # Get the html contents of the webpage
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all form elements in the HTML
            form_elements = soup.find_all('form')

            # Initialize mail function classification
            mail_function_classification = 1  # Default: Legitimate

            # Check each form element for any occurrence of mail() or mailto: in the action attribute
            for form in form_elements:
                action = form.get('action', '')

                # Check if the action attribute contains "mail()" or "mailto:"
                if 'mail()' in action or 'mailto:' in action:
                    mail_function_classification = -1  # Phishing

            # Add mail function classification to the features dictionary with labels
            if mail_function_classification == -1:
                features['Submitting_to_email'] = -1  # Phishing
            else:
                features['Submitting_to_email'] = 1  # Legitimate

        else:
            print(f"Failed to retrieve the website. Status code: {response.status_code}")
            features['Submitting_to_email'] = -1  # Phishing

    except Exception as e:
        print(f"An error occurred: {e}")
        features['Submitting_to_email'] = -1  # Phishing

    # Feature 11: Abnormal URL
    try:
        # Parse the URL to get the hostname
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname

        # Check if the hostname is None or empty
        if not hostname:
            features['Abnormal_URL'] = -1  # Phishing (abnormal URL)
        else:
            features['Abnormal_URL'] = 1  # Legitimate (hostname included in URL)

    except Exception as e:
        print(f"An error occurred: {e}")
        return -1  # Phishing (abnormal URL)
    
    # Feature 12: Number of Redirects
    try:
        # Send a HEAD request to the URL to follow redirects without downloading the content
        response = requests.head(url, allow_redirects=True)

        # Count the number of redirects by examining the history of the response
        num_redirects = len(response.history)

        if num_redirects <=2:
            features["Redirect"] = 1
        elif 2< num_redirects <=4:
            features["Redirect"] = 0
        else:
            features["Redirect"] = -1
        

    except Exception as e:
        # Handle any exceptions that might occur during the request
        print(f"An error occurred: {e}")
        return None
    
    # Feature 13: On mouseover
    try:
        # Send an HTTP GET request to the website
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all elements with the "onMouseOver" event attribute
            elements_with_onmouseover = soup.find_all(attrs={"onMouseOver": True})

            # Check if any of these elements change the status bar
            for element in elements_with_onmouseover:
                if "window.status" in element.get("onMouseOver", ""):
                    features["on_mouseover"] = -1  # Phishing
                    return

            # If no element changes the status bar, classify as Legitimate
            features["on_mouseover"] = 1  # Legitimate
        else:
            print(f"Failed to retrieve the website. Status code: {response.status_code}")
            features["on_mouseover"] = -1  # Phishing
    except Exception as e:
        print(f"An error occurred: {e}")
        features["on_mouseover"] = -1  # 
        
    # Feature 14: Disabled right click
    try:
        # Send an HTTP GET request to the website
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all elements with the "oncontextmenu" event attribute
            elements_with_oncontextmenu = soup.find_all(attrs={"oncontextmenu": True})

            # Check if any of these elements disable right-click
            for element in elements_with_oncontextmenu:
                if "event.button==2" in element.get("oncontextmenu", ""):
                    features["RightClick"] = -1  # Phishing
                    return

            # If no element disables right-click, classify as Legitimate
            features["RightClick"] = 1  # Legitimate
        else:
            print(f"Failed to retrieve the website. Status code: {response.status_code}")
            features["RightClick"] = -1  # Phishing
    except Exception as e:
        print(f"An error occurred: {e}")
        features["RightClick"] = -1  # Phishing
        
    # Feature 15: Iframe
    try:
        # Send an HTTP GET request to the website
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all input elements with type "text" inside pop-up windows (iframes)
            popups_with_text_fields = soup.find_all('iframe')
            contains_text_fields = False

            for popup in popups_with_text_fields:
                # Send an HTTP GET request to the iframe URL
                iframe_url = popup.get('src')
                iframe_response = requests.get(iframe_url)

                # Check if the iframe request was successful (status code 200)
                if iframe_response.status_code == 200:
                    iframe_soup = BeautifulSoup(iframe_response.text, 'html.parser')

                    # Find input elements with type "text" in the iframe
                    text_inputs = iframe_soup.find_all('input', {'type': 'text'})

                    # If text inputs are found, classify as phishing
                    if text_inputs:
                        contains_text_fields = True
                        break

            # If any popup window contains text fields, classify as phishing
            if contains_text_fields:
                features["popUpWidnow"] = -1
                features["Iframe"] = -1   # Phishing
            else:
                features["popUpWidnow"] = 1
                features["Iframe"] = 1   # Legitimate
        else:
            print(f"Failed to retrieve the website. Status code: {response.status_code}")
            features["popUpWidnow"] = -1
            features["Iframe"] = -1   # Phishing
    except Exception as e:
        print(f"An error occurred: {e}")
        features["popUpWidnow"] = -1
        features["Iframe"] = -1   # Phishing

    # Feature 17: Check the age of the domain
    try:
        if "creation_date" in res:
            creation_date = res["creation_date"]
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            creation_date = datetime.strptime(str(creation_date), "%Y-%m-%d %H:%M:%S")
            current_date = datetime.now()
            age_in_months = (current_date.year - creation_date.year) * 12 + (current_date.month - creation_date.month)

            if age_in_months < 6:
                features['age_of_domain'] = -1 
            else:
                features['age_of_domain'] = 1
        else:
            features['age_of_domain'] = -1 


    except Exception as e:
        print("Error performing WHOIS lookup:", str(e))

    # Feature 18: Check for DNS record
    try:
        if "dnssec" not in res:
            features['DNSRecord'] = -1
        else:
            features['DNSRecord'] = 1

    except Exception as e:
        print("Error performing WHOIS lookup:", str(e))

    # Feature 20: Check for google index and stats
    google_index = indexAndStats(url)
    if google_index == True:
        features["Google_Index"] = 1
        features["Statistical_report"] = 1
        # Feature 19: Check for web_traffic,PageRank,backlinks
        web_traffic,PageRank,backlinks = seoStats(url)
        if web_traffic <= 200000:
            features["web_traffic"] = -1
        else:
            features["web_traffic"] = 1
        if PageRank <= 5:
            features["Page_Rank"] = -1
        else:
            features["Page_Rank"] = 1
        if backlinks <= 200:
            features["Links_pointing_to_page"] = -1
        else:
            features["Links_pointing_to_page"] = 1
    else:
        features["Google_Index"] = -1
        features["Statistical_report"] = -1
        features["web_traffic"] = -1
        features["Page_Rank"] = -1
        features["Links_pointing_to_page"] = -1

    # Returing Features
    return features

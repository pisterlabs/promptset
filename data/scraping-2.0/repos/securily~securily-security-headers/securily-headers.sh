#!/usr/bin/env python3
import time
import requests
import json
import re
import datetime
import os
import argparse

# Path to the configuration file
web_config_file_path = "web_configuration.json"
app_config_file_path = "app_configuration.json"
api_config_file_path = "api_configuration.json"

# Configure argparse for command-line arguments
parser = argparse.ArgumentParser(description="Security Security Headers Open Source Scanner powered by OpenAI")
parser.add_argument("-u", "--url-to-scan", type=str, required=True, help="URL to scan")
parser.add_argument("-o", "--openai-api-key", type=str, required=True, help="OpenAI API Key")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
parser.add_argument("-f", "--force-reload", action="store_true", help="Force reload the configuration")
parser.add_argument("-a", "--authorization-for-api", type=str, help="Authorization Token For APIs")
args = parser.parse_args()

# The OpenAI API Key
OPENAI_API_KEY = args.openai_api_key
# The URL to Scan
URL = args.url_to_scan
# The Authorization Token for an API Scan
authorization = args.authorization_for_api

# List of HTTP headers
headers_to_read = []
# Array to store configuration
configuration = []

web_security_headers = [
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "X-XSS-Protection",
    "Referrer-Policy",
    "Permissions-Policy"
]

app_security_headers = [
    "Content-Type",
    "Content-Length",
    "Cache-Control"
]

api_security_headers = [
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "Strict-Transport-Security",
    "X-Frame-Options",
    "Access-Control-Allow-Origin",
    "Access-Control-Allow-Methods",
    "Access-Control-Allow-Headers"
]


def normalizeUrl(url):
    if not re.match('^https?://', url):
        url = 'https://' + url
    return url

def fix_bad_json(json_string):
    try:
        # Try loading the JSON string as is
        json_data = json.loads(json_string)
        return json_data

    except json.JSONDecodeError as e:
        if args.verbose:
          print("Error occurred while decoding JSON: {}".format(str(e)))

        # Remove the invalid control character
        cleaned_json = ''.join(c for c in json_string if c.isprintable())
        try:
            json_data = json.loads(cleaned_json)
            if args.verbose:
              print("Successfully fixed the bad JSON.")
            return json_data

        except json.JSONDecodeError as e:
            if args.verbose:
              print("Error occurred while fixing the bad JSON: {}".format(str(e)))

    return None


def read_headers_from_url(url, test_type='web', authorization=None, payload={}):
    try:
        url = normalizeUrl(url)
        if test_type == 'web' or test_type == 'app':
            payload = {}
            request_headers = {}

            response = requests.request("GET", url, headers=request_headers, data=payload, allow_redirects=True)
        elif test_type == 'api':
            request_headers = {
                'Authorization': 'Bearer ' + authorization
            }
            response = requests.request("GET", url, headers=request_headers, data=payload, allow_redirects=True)
        else:
            response = requests.head(url, allow_redirects=True)

        response_headers = response.headers

        headers = {}
        status_code = response.status_code

        for header in headers_to_read:
            header_value = response_headers.get(header)
            if header_value and (header_value.startswith("http://") or header_value.startswith("https://")):
                try:
                    response = requests.get(header_value, verify=False)
                    if response.status_code == 200:
                        header_value = response.text
                        headers[header] = header_value
                    else:
                        header_value = "Failed to download content"
                except requests.exceptions.RequestException as e:
                    header_value = "Failed to download content: {}".format(str(e))

            if header_value:
                headers[header] = header_value
            else:
                headers[header] = "Not Found"

        return headers, status_code

    except requests.RequestException as e:
        if args.verbose:
          print("Error occurred while reading headers from URL: {}".format(str(e)))
        return None


def compare_headers_configuration(headers_found, source_configuration, status_code, url):
    results = []

    for header in headers_to_read:
        header_config = next((config for config in source_configuration if config['name'] == header), None)

        header_value = headers_found.get(header)

        if header_config:
            if header.lower() == header_config.get('name').lower() and (header_config.get('values') and any(keyword.lower() in header_value.lower() for keyword in header_config['values'].split(', '))):
                results.append({"url": url, "name": header_config.get('name'), "value": (header_value or "n/a"), "severity": header_config.get('severity'), "reason": header_config.get('reason'), "remediation": header_config.get('remediation'), "values": header_config.get('values'), "status": "PASS", "status_code": status_code})
            elif header.lower() == header_config.get('name').lower() and (header_value != 'Not Found'):
                results.append({"url": url, "name": header_config.get('name'), "value": (header_value or "n/a"), "severity": header_config.get('severity'), "reason": header_config.get('reason'), "remediation": header_config.get('remediation'), "values": header_config.get('values'), "status": "PASS", "type": "EXCELLENT", "status_code": status_code})
            else:
                results.append({"url": url, "name": header_config.get('name'), "value": (header_value or "n/a"), "severity": header_config.get('severity'), "reason": header_config.get('reason'), "remediation": header_config.get('remediation'), "values": header_config.get('values'), "directives": header_config.get('directives'), "status": "FAIL", "status_code": status_code})
        else:
            results.append({"url": url, "name": header, "value": (header_value or "n/a"), "severity": "", "reason": "", "remediation": "", "values": "", "directives": "", "status": "FAIL", "status_code": status_code})

    return results


def configure_headers(headers_to_configure, path_to_configure):
    # Define the array to store configurations
    configuration = []

    # Iterate over each header
    for header in headers_to_configure:
        retries = 5  # Number of retries
        while retries > 0:
            # Define the prompt
            prompt = 'For the following security header: {}, return a JSON object in the following format: {{"name": "Header Name", "severity": "Severity Rating", ' \
                     '"reason": "Explain for a non technical person justifying the severity classification of the finding", "remediation": "Step-by-step instructions for remediation.", ' \
                     '"values": "possible values for the content security policy header comma separated", "directives": "header directives with examples"}}'.format(header)

            # Create the JSON data for the request
            json_data = {
                "prompt": prompt,
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 1,
                "n": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }

            # Send the request to OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/engines/text-davinci-003/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(OPENAI_API_KEY)
                },
                json=json_data
            )

            try:
                
                print("Working on header {}".format(header))
                # Remove invalid control characters from response
                response_text = response.text
                cleaned_response = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', response_text)

                # Extract JSON object from cleaned response
                extracted_json = json.loads(cleaned_response)["choices"][0]["text"]

                if extracted_json is not None:
                    try:
                        if args.verbose:
                            print(extracted_json)
                        # Convert the extracted JSON to a dictionary
                        extracted_dict = fix_bad_json(extracted_json)

                        if args.verbose:
                            print(extracted_dict)

                        # Retry the OpenAPI call if the extracted dictionary is null
                        if extracted_dict is None:
                            print("Failed to extract non-null value for header '{}'. Retrying the OpenAPI call...".format(header))
                            continue
                        # Append the extracted dictionary to the array
                        configuration.append(extracted_dict)
                        print("Successfully processed header {}".format(header))

                        # Print the extracted dictionary
                        if args.verbose:
                            print("Extracted Configuration for '{}':".format(header))
                            print(extracted_dict)
                            print()

                        # Break the retry loop if the extraction is successful
                        break

                    except json.JSONDecodeError as e:
                        if args.verbose:
                            print("Error occurred while decoding JSON for header '{}': {}".format(header, str(e)))
                        retries -= 1  # Decrease the number of retries

                else:
                    if args.verbose:
                        print("Received null value for header '{}'. Retrying...".format(header))
                    retries -= 1  # Decrease the number of retries

            except (json.JSONDecodeError, IndexError, KeyError) as e:
                if args.verbose:
                    print("Error occurred while extracting JSON for header '{}': {}".format(header, str(e)))
                retries -= 1  # Decrease the number of retries

            # Sleep for a moment before retrying
            time.sleep(1)

    # Save the configuration to a file
    with open(path_to_configure, "w") as file:
        json.dump(configuration, file)

# Determine the type of headers based on the URL
URL = normalizeUrl(URL)
testType = ''
if "app" in URL:
    headers_to_read = web_security_headers + app_security_headers
    config_file_path = app_config_file_path
    testType = 'app'
elif "api" in URL:
    # Check if 'authorization' exists and is not None and not empty
    if 'authorization' in locals() and authorization is not None and authorization.strip() != "":
        headers_to_read = api_security_headers
        config_file_path = api_config_file_path
        testType = 'api'
    else:
        # Handle the case where 'authorization' is missing or empty
        headers_to_read = web_security_headers
        config_file_path = web_config_file_path
        testType = 'web'
else:
    headers_to_read = web_security_headers
    config_file_path = web_config_file_path
    testType = 'web'

# Example usage of read_headers_from_url()
status_code = 0
headers, status_code = read_headers_from_url(URL, testType, authorization)
if headers:
    print("Finished reading headers from URL: {}".format(URL))
    if args.verbose:
        print("Headers from URL: {}".format(URL))
        print(json.dumps(headers, indent=4))
        print()

# Configure headers using OpenAI API
if args.force_reload or not os.path.exists(config_file_path):
    # Call the configure_headers() function to create or reload the configuration file
    print("Loading headers from OpenAI, this will take a while")
    configure_headers(headers_to_read, config_file_path)
    
    with open(config_file_path, "r") as file:
        configuration = json.load(file)
    print("Finished configuring headers using OpenAI API")
else:
    # Get the modification time of the configuration file
    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(config_file_path))

    # Calculate the time difference between now and the modification time
    time_diff = datetime.datetime.now() - mod_time

    # Check if the configuration file is older than or equal to 30 days
    if time_diff.days >= 30:
        # Call the configure_headers() function to update the configuration
        print("Loading headers from OpenAI, this will take a while")
        configure_headers(headers_to_read, config_file_path)
    else:
        # Load the configuration from the file
        with open(config_file_path, "r") as file:
            configuration = json.load(file)
        
        print("Finished loading existing configuration from file")

# Compare headers and configuration
results = compare_headers_configuration(headers, configuration, status_code, URL)

print("Finished comparing headers and configuration")

# Store the results in a list
results_list = []
for result in results:
    results_list.append(result)

# Print the results in JSON format
results_json = json.dumps(results_list, indent=4)
print(results_json)

# Write the results to a file
with open("results.json", "w") as file:
    file.write(results_json)

print("Results have been saved to results.json")
import requests
import base64
import time
from bs4 import BeautifulSoup
import openai
import webbrowser
import os
import re
import sys

openai.api_key = "UPDATE"

def list_emails(oauth_token):
    # Set up the API endpoint
    url = "https://gmail.googleapis.com/gmail/v1/users/me/messages"

    # Prepare headers for the request
    headers = {
        "Authorization": f"Bearer {oauth_token}",
    }

    # Prepare the parameters for the request (in this case, max results)
    params = {
        "maxResults": 1,  # number of emails to list
    }

    # Make the GET request to the Gmail API
    response = requests.get(url, headers=headers, params=params)

    # If the request is successful
    if response.status_code == 200:
        # Parse the JSON response
        emails = response.json()
        return emails
    else:
        return None


# Function to get the full email details
def get_email_details(oauth_token, email_id):
    # Set up the API endpoint
    url = f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{email_id}?format=full"

    # Prepare headers for the request
    headers = {
        "Authorization": f"Bearer {oauth_token}",
    }

    # Make the GET request to the Gmail API
    response = requests.get(url, headers=headers)

    # If the request is successful
    if response.status_code == 200:
        return response.json()
    else:
        return None


# Function to decode email parts
def decode_email_parts(parts):
    email_body = ""
    # parts.pop(0)
    if parts:
        for part in parts:
            if part['mimeType'] == 'text/html' and 'data' in part['body']:
                email_body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            elif 'parts' in part:
                email_body += decode_email_parts(part['parts'])
    # print(email_body)
    return email_body


# Function to parse the email details for keywords and return the count
def parse_email_for_keywords(email_data, keywords):
    email_body = decode_email_parts(email_data.get('payload', {}).get('parts', []))
    return sum(keyword in email_body for keyword in keywords), email_body


# Function to find the confirmation link in the email body
def gen_links_list(email_body):
    # parser = LinkParser()
    # parser.feed(email_body)
    # return getattr(parser, "link", None)
    links = []
    soup = BeautifulSoup(email_body, 'html.parser')
    all_links = soup.find_all('a')
    for link in all_links:
        href = link.get('href')
        if href and "mailto" not in href:
            links.append(href)
    return links

    # Function to check if a URL is a verification link


def is_verification_link(urls,email_body):
    # Construct the prompt
    prompt = "Determine if the following email has a verification link, and if so, what it is:";
    prompt += email_body
    prompt += "Desired format: Yes/no, the verification link is: "

    # Call the OpenAI API
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300,
    )
    # print(response)
    output_text = response.choices[0].text.strip()
    # print("Output text" + output_text)

    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    for line in output_text.split('\n'):
        # print(line)
        if re.search(url_pattern, line):
            url = line.split()[-1]
            #print("URL:" + url)
            return url
    return "None"







# Function to periodically check emails for keywords
def check_emails_for_keywords(oauth_token, check_interval, timeout, keywords):
    start_time = time.time()
    best_match = None
    most_keywords_found = 0

    while True:
        emails = list_emails(oauth_token)
        if emails is not None:
            for email in emails["messages"]:
                email_data = get_email_details(oauth_token, email['id'])
                if email_data:
                    keyword_count, email_body = parse_email_for_keywords(email_data, keywords)
                    if keyword_count > most_keywords_found:
                        most_keywords_found = keyword_count
                        best_match = email_data
                        confirmation_links = gen_links_list(email_body)
                        # print(confirmation_links)

                        # If we have found a good match, return the details
                        if most_keywords_found > 0:
                            headers = best_match['payload']['headers']
                            subject = next(header['value'] for header in headers if header['name'].lower() == 'subject')
                            time2 = next(header['value'] for header in headers if header['name'].lower() == 'received')
                            # print(time2)
                            from_email = next(header['value'] for header in headers if header['name'].lower() == 'from')

                            return {
                                "email_address": from_email,
                                "subject": subject,
                                "confirmation_link": is_verification_link(confirmation_links,email_body)

                            }

        # Check if the timeout has been reached
        if time.time() - start_time > timeout:
            return best_match  # Return the best match we found before timeout

        # Wait for the specified interval before checking again
        time.sleep(check_interval)


# Usage

# Your OpenAI API Key, ensure this is kept secure

oauth_token = sys.argv[1]  # Replace with your actual OAuth token
check_interval = 60  # seconds
timeout = 3600  # 1 hour, for example
keywords = ["confirmation", "verify", "subscribe", "candidate", "email address", "check", "grant", "access"]  # Keywords to look for

email_details = check_emails_for_keywords(oauth_token, check_interval, timeout, keywords)
# print(email_details)
# if email_details:
    # print(f"Email address: {email_details['email_address']}")
    # print(f"Subject: {email_details['subject']}")
    # print(f"Confirmation link: {email_details['confirmation_link']}")
    # if(email_details['confirmation_link'] != "None"):
    #     webbrowser.open_new_tab(email_details['confirmation_link'])
# else:
    # print("No email was found with the specified keywords within the timeout period.")

if email_details['confirmation_link'] == "None":
    print(1)
else:
    print(email_details['confirmation_link'])

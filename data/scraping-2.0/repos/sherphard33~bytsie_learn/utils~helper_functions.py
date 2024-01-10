import os
import ssl
import json
import socket
import openai
import smtplib
import requests


from requests_oauthlib import OAuth1Session
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


openai.api_key = os.getenv('openai_api_key')
ig_user_id = os.getenv('instagram_business_account')
inst_access_token = os.getenv('meta_access_token')
meta_access_token = os.getenv('meta_access_token')
meta_client_id = os.getenv('meta_client_id')
meta_client_secret = os.getenv('meta_client_secret')
meta_business_id = os.getenv('meta_business_id')
meta_page_id = os.getenv('meta_page_id')
instagram_business_account = os.getenv('instagram_business_account')

twitter_app_id = os.getenv("twitter_app_id")
twitter_bearer_token = os.getenv("twitter_bearer_token")
twitter_consumer_key = os.getenv("twitter_api_key")
twitter_consumer_secret = os.getenv("twitter_api_secret")


def save_data(data, filename, type):
    with open(f'{filename}_{type}.doc', 'w') as f:
        f.write(data)


def basic_generation(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def search_engine(user_prompt):
    search_result = openai.Engine("davinci").search(
        documents=[user_prompt],
        query="What is the fastest way to make money online?",
        max_rerank=10,
        return_metadata=True,
    )
    return search_result


def generate_thumbnails(user_prompt: str = "A cute baby sea otter"):
    results = openai.Image.create(
        prompt=user_prompt,
        n=2,
        size="1024x1024"
    )
    print(results)
    return results


def facebook_post(caption: str):

    # Post Content as Text
    post_url = 'https://graph.facebook.com/{}/feed'.format(meta_page_id)
    payload = {
        'message': caption,
        'access_token': meta_access_token
    }
    r = requests.post(post_url, data=payload)
    print(r.text)


def insta_post(caption: str, image_location: str):
    # Post the Image to Instagram
    post_url = 'https://graph.facebook.com/v16.0/{}/media'.format(ig_user_id)
    payload = {
        'image_url': image_location,
        'caption': caption,
        'access_token': inst_access_token
    }
    r = requests.post(post_url, data=payload)
    print(r.text)
    result = json.loads(r.text)
    if 'id' in result:
        creation_id = result['id']
        second_url = 'https://graph.facebook.com/v16.0/{}/media_publish'.format(
            ig_user_id)
        second_payload = {
            'creation_id': creation_id,
            'access_token': inst_access_token
        }
        r = requests.post(second_url, data=second_payload)
        print('--------Just posted to instagram--------')
        print(r.text)
    else:
        print('HOUSTON we have a problem')


def twitter_post(payload):

    # Get request token
    request_token_url = "https://api.twitter.com/oauth/request_token?oauth_callback=oob&x_auth_access_type=write"
    oauth = OAuth1Session(twitter_consumer_key,
                          client_secret=twitter_consumer_secret)

    try:
        fetch_response = oauth.fetch_request_token(request_token_url)
    except ValueError:
        print(
            "There may have been an issue with the consumer_key or consumer_secret you entered."
        )

    resource_owner_key = fetch_response.get("oauth_token")
    resource_owner_secret = fetch_response.get("oauth_token_secret")
    print("Got OAuth token: %s" % resource_owner_key)

    # Get authorization
    base_authorization_url = "https://api.twitter.com/oauth/authorize"
    authorization_url = oauth.authorization_url(base_authorization_url)
    print("Please go here and authorize: %s" % authorization_url)
    verifier = input("Paste the PIN here: ")

    # Get the access token
    access_token_url = "https://api.twitter.com/oauth/access_token"
    oauth = OAuth1Session(
        twitter_consumer_key,
        client_secret=twitter_consumer_secret,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )
    oauth_tokens = oauth.fetch_access_token(access_token_url)

    access_token = oauth_tokens["oauth_token"]
    access_token_secret = oauth_tokens["oauth_token_secret"]

    # Make the request
    oauth = OAuth1Session(
        twitter_consumer_key,
        client_secret=twitter_consumer_secret,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret,
    )

    # Making the request
    response = oauth.post(
        "https://api.twitter.com/2/tweets",
        json=payload,
    )

    if response.status_code != 201:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text)
        )

    print("Response code: {}".format(response.status_code))

    # Saving the response as JSON
    json_response = response.json()
    print(json.dumps(json_response, indent=4, sort_keys=True))

    return json_response


def send_email():
    # Set up the SMTP server
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = "your_email@gmail.com"
    password = input("Type your password and press enter: ")

    # Create a secure connection with the server and start TLS
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls(context=context)
        server.login(sender_email, password)

        # Send email
        receiver_email = "recipient_email@gmail.com"
        message = """\
        Subject: Test Email

        This is a test email sent from Python."""

        server.sendmail(sender_email, receiver_email, message)

from dotenv import load_dotenv
from twilio.rest import Client

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException

import string
import os
import openai
import time

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

linkedin_user = os.environ.get("LINKEDIN_USER")
linkedin_pass = os.environ.get("LINKEDIN_PASS")

twilio_sid = os.environ.get("TWILIO_SID")
twilio_auth = os.environ.get("TWILIO_AUTH")
twilio_client = Client(twilio_sid, twilio_auth)

MY_PHONE = "whatsapp:+18329517889"
TWILIO_NUMBER = "whatsapp:+14155238886"

TWILIO_RETRY_TIME = 5
FIRST_ELEMENT = 0

OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
LINKEDIN_SITE = 'https://www.linkedin.com'


def post_to_linkedin(contents):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    # driver = webdriver.Chrome()
    driver.get(LINKEDIN_SITE)
    time.sleep(5)

    try:
        driver.find_element(By.XPATH, '//*[@id="session_key"]').send_keys(linkedin_user)
        time.sleep(1)
        driver.find_element(By.XPATH, '//*[@id="session_password"]').send_keys(linkedin_pass)
        time.sleep(1)
        driver.find_element(By.CLASS_NAME, 'sign-in-form__submit-btn--full-width').click()
        time.sleep(5)
        driver.find_element(By.CLASS_NAME, 'share-box-feed-entry__trigger').click()
        time.sleep(1)
        driver.find_element(By.CLASS_NAME, 'ql-editor.ql-blank').send_keys(contents)
        time.sleep(10)
        driver.find_element(By.CLASS_NAME, 'share-actions__primary-action').click()
        time.sleep(5)
    except NoSuchElementException as err:
        print("Issue finding an element, please try again")
    except ElementClickInterceptedException as err:
        print("Click was intercepted, contact Tim for further diagnoses")
    else:
        return


def verify_response():
    received_history = twilio_client.messages.list(to=TWILIO_NUMBER, from_=MY_PHONE)
    sent_history = twilio_client.messages.list(to=MY_PHONE, from_=TWILIO_NUMBER)

    if not received_history:
        time.sleep(TWILIO_RETRY_TIME)
        return verify_response()

    latest_message_from_tim = received_history[FIRST_ELEMENT]
    latest_message_from_twilio = sent_history[FIRST_ELEMENT]
    tim_time_sent = latest_message_from_tim.date_sent
    twilio_time_sent = latest_message_from_twilio.date_sent

    if twilio_time_sent <= tim_time_sent:
        if latest_message_from_tim.body.lower() == "y" or latest_message_from_tim.body.lower() == "yes":
            return True
        else:
            print("Quote not good, exiting verification process")
            return False
    else:
        print("No response yet, waiting for input")
        time.sleep(TWILIO_RETRY_TIME)
        return verify_response()


def get_valid_quote():
    prompt = "Generate a simple random topic or event in 7 words or less"
    query = [{"role": "user", "content": prompt}]
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=query
    )

    reply = chat.choices[FIRST_ELEMENT].message.content
    quote_prompt = "Generate a single sentence, simple quote with about {} in 10 words or less".format(reply)
    quote_query = [{"role": "user", "content": quote_prompt}]

    quote_chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=quote_query
    )

    end_quote = quote_chat.choices[FIRST_ELEMENT].message.content.translate(str.maketrans('', '', string.punctuation))

    sting_prompt = "Generate 2 words relating to {} with no punctuation".format(reply)
    sting_query = [{"role": "user", "content": sting_prompt}]

    sting_chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=sting_query
    )

    sting_result = sting_chat.choices[FIRST_ELEMENT].message.content.lower().translate(str.maketrans('', '', string.punctuation))

    linkedin = "Wow! {} - {}!".format(end_quote, sting_result)
    print(linkedin)

    message = twilio_client.messages.create(
        to=MY_PHONE,
        from_=TWILIO_NUMBER,
        body="Quote: {}\nIs this good?".format(linkedin))

    valid = verify_response()

    if valid:
        print("Good quote, posting to LinkedIn!")
        return linkedin
    else:
        print("Bad quote, regenerating!")
        return get_valid_quote()


def main():
    quote = get_valid_quote()
    print("posting to linkedin now this: {}".format(quote))
    # post_to_linkedin(quote)


if __name__ == "__main__":
    main()

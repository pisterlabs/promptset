import requests
from bs4 import BeautifulSoup
import re
import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_data_from_website(landing_page_url):
    # URL of the web page to scrape
    if not landing_page_url:
        print("Please provide a landing page")
        return ""
    webscraping_landing_page = ""
    try:
        # Send a GET request to the URL
        response = requests.get(landing_page_url)

        # Create a BeautifulSoup object with the response content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the body element
        body = soup.find("body")

        # Extract the text from the body element and remove line breaks
        body_text = body.get_text()
        # Remove consecutive line breaks  # Remove consecutive line breaks
        body_text = remove_extra_spaces(body_text)

        webscraping_landing_page_prompt = f"""Write a summary of 250 words or less of the key details about the 
        company mentioned in the paragraph below. Provide information about the company's name, description, 
        industry, products/services and more relevant information you consider. {body_text}"""
        context_messages = [
            {
                "role": "system",
                "content": "You are an AI that summarizes and finds relevant information of companies from their website",
            },
            {"role": "user", "content": webscraping_landing_page_prompt},
        ]

        webscraping_landing_page = get_completion_from_messages(context_messages)
    except Exception as e:
        print(e)
        webscraping_landing_page = ""
    return webscraping_landing_page


def remove_extra_spaces(text):
    # Replace multiple whitespaces (excluding line breaks) with a single whitespace
    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text.strip()


print(get_data_from_website("https://www.skandia.co/"))

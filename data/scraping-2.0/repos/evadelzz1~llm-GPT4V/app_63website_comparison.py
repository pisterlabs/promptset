from dotenv import load_dotenv
from openai import OpenAI
import base64
import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Initialize the OpenAI client with the API key
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Selenium WebDriver
options = Options()
options.headless = True
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

def get_website_snapshot(url):
    driver.get(url)
    time.sleep(3)
    screenshot = driver.get_screenshot_as_png()
    return base64.b64encode(screenshot).decode("utf-8")

# URLs of the websites to compare
website_url_1 = "https://en.wikipedia.org/wiki/Pyramid"
website_url_2 = "https://en.wikipedia.org/wiki/Artificial_intelligence"

# Take snapshots of both websites
snapshot_1 = get_website_snapshot(website_url_1)
snapshot_2 = get_website_snapshot(website_url_2)

driver.quit()

# Prepare the prompt for the AI model
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "Can you create a sci-fi story based on these two images?",
            {"image": snapshot_1},
            {"image": snapshot_2},
        ],
    },
]

params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 1000,
    "stream": True,    
}

# Get the response from OpenAI API
response = client.chat.completions.create(**params)

full_response = ""
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
        full_response += str(chunk.choices[0].delta.content)

# Output the result
# print(full_response)
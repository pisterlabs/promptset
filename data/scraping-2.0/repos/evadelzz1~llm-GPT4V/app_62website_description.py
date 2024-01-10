from dotenv import load_dotenv
from openai import OpenAI
import base64
import os

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

# Go to the website and take a screenshot
driver.get("https://example.com/")
screenshot = driver.get_screenshot_as_png()
driver.quit()

# Encode the screenshot in base64
screenshot_base64 = base64.b64encode(screenshot)
        
# Decode byte string into UTF-8 to get a string representation of base64
screenshot_base = screenshot_base64.decode("utf-8")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "This is a snapshot of a website."
            "Provide a detailed description of its current design "
            "and suggest three improvements that could be made to the layout, color schema, and overall user experience.",
            {"image": screenshot_base},
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
# description_and_suggestions = response.choices[0].message.content:

full_response = ""
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
        full_response += str(chunk.choices[0].delta.content)


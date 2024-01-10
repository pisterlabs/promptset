import os
import random
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth ## https://pypi.org/project/selenium-stealth/
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import openai


def generate_review(product_name, is_positive=True):
    prompt = f"As an AI, I am reviewing the {product_name}."
    if is_positive:
        prompt += " It is an excellent product because"
    else:
        prompt += " It is a disappointing product because"

     # Set up OpenAI API credentials
    openai.api_key = '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'

    # Generate review
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None
    )

    review = response.choices[0].text.strip().replace("\n", "")
    return review


options = webdriver.ChromeOptions()
options.add_argument("start-maximized")

# options.add_argument("--headless")

options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)

s = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=s, options=options)
stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
        )

driver.get('http://localhost:8501/')
time.sleep(3)

# login and password inputs:
input = driver.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[2]/div/div[1]/div/input')

for i in range(1,10):
    pn = random.choice([True, False])
    print(pn)
    
    if pn == True:
        # Generate a positive review for a product
        positive_review = generate_review("AmazingProduct", is_positive=True)

        print("Positive Review:")
        print(positive_review)

        input.send_keys(f"{positive_review}")
        time.sleep(3)

        # click on the button to login
        button_enter = driver.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[3]/div/button/div/p')
        button_enter.click()

        driver.clear()
        time.sleep(2)

    else:
        # Generate a negative review for a product
        negative_review = generate_review("DisappointingProduct", is_positive=False)
        print("Negative Review:")
        print(negative_review)

        input.send_keys(f"{negative_review}")
        time.sleep(3)
        
        # click on the button to login
        button_enter = driver.find_element('xpath','//*[@id="root"]/div[1]/div[1]/div/div/div/section/div[1]/div[1]/div/div[3]/div/button/div/p')
        button_enter.click()
        
        driver.clear()
        time.sleep(2)

#driver.close()
import unittest
from appium import webdriver
from appium.webdriver.common.mobileby import MobileBy as AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
from bs4 import BeautifulSoup
import time
import openai
openai.api_key = "INSERT YOUR API KEY HERE"
from appium import webdriver
from appium.webdriver.common.mobileby import MobileBy as AppiumBy
import openai


desired_caps = {
    'platformName': 'Android',
    'deviceName': 'R5CT6057W7Y',
    'noReset': True,
    'appPackage': 'com.instagram.barcelona',
    'appActivity': 'com.instagram.barcelona.mainactivity.BarcelonaActivity',
    'automationName': 'UiAutomator2'
}

class MyApplication:
    def __init__(self):
        self.driver = webdriver.Remote('http://localhost:4726/wd/hub', desired_caps)
    def close(self):
        if self.driver:
            self.driver.quit()

    def find_and_click_create_element(self):
        elements = self.driver.find_elements(AppiumBy.XPATH, '//*')
        create_elements = [element for element in elements if element.tag_name == 'Create']
        if create_elements:
            create_element = create_elements[0]
            print(
                f'Clicking on Element - Tag: {create_element.tag_name}, Text: {create_element.text}, Attributes: {create_element.get_attribute("class")}')
            create_element.click()
        else:
            print("No 'Create' element found")

    def send_text(self, text):
        shift_on = False
        for char in text:
            if char.isupper():
                if not shift_on:
                    self.driver.press_keycode(59)
                    shift_on = True
                keycode = ord(char.lower()) - ord('a') + 29
            elif char.islower():
                if shift_on:
                    self.driver.press_keycode(59)
                    shift_on = False
                keycode = ord(char) - ord('a') + 29
            elif char.isspace():
                keycode = 62
            elif char == '!':
                if not shift_on:
                    self.driver.press_keycode(59)
                    shift_on = True
                keycode = 8
            else:
                continue

            self.driver.press_keycode(keycode)

        if shift_on:
            self.driver.press_keycode(59)

    def find_and_click_post_element(self):
        elements = self.driver.find_elements(AppiumBy.XPATH, '//*')
        post_elements = [element for element in elements if element.text == 'Post']
        if post_elements:
            post_element = post_elements[0]
            print(
                f'Clicking on Element - Tag: {post_element.tag_name}, Text: {post_element.text}, Attributes: {post_element.get_attribute("class")}')
            post_element.click()
        else:
            print("No 'Post' element found")



def get_reddit_titles():
    url = 'https://www.reddit.com/r/worldnews/'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    title_elements = soup.find_all('h3', class_='_eYtD2XCVieq6emjKBH3m')

    titles = [element.text for element in title_elements]
    return titles
def gpt35Predict(prompt, retries=3, delay=1):
    for attempt in range(retries):
        try:
            # Send an API request and get a response, note that the interface and parameters have changed compared to the old model
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": f"{prompt}"}],
                max_tokens=256,
            )
            output_text = response['choices'][0]['message']['content']
            return output_text
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:  # no delay on the last attempt
                time.sleep(delay)
            else:
                raise  # re-raise the last exception if all attempts failed


def provide_context():
    titles = get_reddit_titles()
    prompt= f"{titles[3]}: please provide some background context. It doesnt have to be up to date. only return a numbered list of 3 short sentences, 10 words max each."
    response = gpt35Predict(prompt)
    fullResponse = f"{titles[3]}:\n{response}"
    return fullResponse
    
if __name__ == '__main__':
    message = provide_context()
    print(message)
    app = MyApplication()
    app.find_and_click_create_element()
    app.send_text(message)
    app.find_and_click_post_element()
    app.close()




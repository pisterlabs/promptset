# Description: This script automates the process of answering a question on Quora using GPT-3.
# Author: harmindesinghnijjar
# Date: 2023-02-24
# Version: 1.0.0
# Usage: python quora_automation.py

# Import modules.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from time import sleep
from webdriver_manager.chrome import ChromeDriverManager
import getpass
import openai
import os
import set_openai_api_key
import time

# Set OpenAI API key.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set username.
user = getpass.getuser()

# Define a Selenium class to automate the browser.
class Selenium(webdriver.Chrome):
    # Constructor method. 
    def __init__(self, webdriverval):
        driver = webdriverval


    # Method to post answer received from GPT-3 on Quora and selecting a personalized answer credential.
    def answer_quora_question(self, answer):
        try:
            
            # Click on button to open answer pop-up.
            answer_popup = self.find_element(By.CSS_SELECTOR, '#mainContent > div > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child(1) > div > div > div > div > div > div > div > div.q-box.qu-zIndex--action_bar > div > div > div:nth-child(1) > button.q-click-wrapper.qu-active--textDecoration--none.qu-focus--textDecoration--none.qu-borderRadius--pill.qu-alignItems--center.qu-justifyContent--center.qu-whiteSpace--nowrap.qu-userSelect--none.qu-display--inline-flex.qu-tapHighlight--white.qu-textAlign--center.qu-cursor--pointer.qu-hover--textDecoration--none.qu-hover--bg--darken.ClickWrapper___StyledClickWrapperBox-zoqi4f-0.iyYUZT.base___StyledClickWrapper-lx6eke-1.fJHGyh')
            answer_popup.click()
        except Exception as e:
            print("Error while clicking on answer popup button: ", e)
            return False
        
        try:
            
            # Paste answer into text box.
            answer_box = self.find_element(By.CSS_SELECTOR, '#root > div > div:nth-child(2) > div > div > div > div > div.q-flex.ModalContainerInternal___StyledFlex-s8es4q-2.gXhqYs.modal_content_inner.qu-flexDirection--column.qu-bg--white.qu-overflowY--auto.qu-borderAll.qu-alignSelf--center > div > div.q-flex.qu-flexDirection--column.qu-overflowY--auto > div.q-relative.qu-display--flex.qu-flexDirection--column > div > div.q-box > div:nth-child(2) > div > div > div > div > div.q-box > div')
            if not answer_box:
                raise ValueError("Could not find answer box element on the page.")
            else:
                answer_box.send_keys(answer)
                print(answer)
        except Exception as e:
            print("Error while pasting answer into text box: ", e)
            print("Unable to answer question, skipping to next question.")
            # Go to next question by calling the answer_quora_question method again.
            self.answer_quora_question(answer)
        
        # Pause for 10 seconds.
        time.sleep(10)
        
        try:
            
            # Post answer.
            post = self.find_element(By.XPATH, '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div[2]/div[2]/div/div[2]/button')
            post.click()
        except Exception as e:
            print("Error while posting answer: ", e)
            return False
        
        # Pause for 2 seconds.
        time.sleep(2)
        
        try:
            
            # Select "I help people with technology and enjoy learning about ML" as your answer credential.
            first_answer_credential = self.find_element(By.XPATH, '//*[@id="root"]/div/div[2]/div/div/div/div/div[2]/div/div[2]/div[1]/div/div[1]/div[3]/div[1]/div/label/div/div[2]/div/span/span[2]')
            
            # Click on the first answer credential element.
            first_answer_credential.click()
        except Exception as e:
            print("Error while selecting answer credential: ", e)
            return False
        
        # Pause for 2 seconds.
        time.sleep(2)
        
        return True

    # Method to extract first question listed on www.quora.com/answers and return it as a string.
    def extract_question(self):
        try:
            question = self.find_element(By.XPATH, '//*[@id="mainContent"]/div/div/div[1]/div/div[2]/div[1]/div/div/div/div/div/div/div/div[1]/div[1]/span/span/a/div/div/div/div/span/span').text
            print(question)
            return question
        except NoSuchElementException:
            print("Could not find question element on the page.")
            return ""
        except Exception as e:
            print("An error occurred while extracting the question:", e)
            return ""
        
    # Method to ask GPT-3 the question extracted from Quora and return the respone text as a string.
    def gpt3_completion(prompt, engine='text-davinci-003', temp=0.7, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0):
        try:
            if not prompt:
                raise ValueError("Prompt cannot be empty.")
            prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen
            )
            if not response.choices:
                raise ValueError("No response from OpenAI API.")
            text = response.choices[0].text.strip()
            if not text:
                raise ValueError("Response text cannot be empty.")
            return text
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
    
    # Method to open Quora in a Chrome instance. 
    def open_quora_questions(self):
        try:
            
            # Open a webpage.
            self.get("https://www.quora.com/answer")
        except Exception as e:
            print("An error occurred while opening Quora:", e)
        else:
            print("Selecting a question from Quora...")
    
    # Method to close the Chrome instance.
    def close_browser(driver):
        """
        Closes the web browser.

        Args:
        driver (selenium.webdriver.Chrome): The Chrome web driver object.

        Returns:
        None
        """
        driver.quit()

    # Method to set up the Chrome instance. 
    def setup(self):
        # AttributeError: 'Selenium' object has no attribute 'w3c'
        # Set Chrome window size to maximize.
        self.maximize_window()
        
        # Set implicit wait time to 10 seconds.
        self.implicitly_wait(10)
        
        # Set page load timeout to 50 seconds.
        self.set_page_load_timeout(50)


 
if __name__ == '__main__':
    # Check if the OpenAI API key is set in the environment using the set_openai_api_key module.
    # If it is, then run the script.
        if set_openai_api_key.check_openai_api_key():
            for i in range(10):
                conversation = list()
                os_command = 'taskkill /im chrome.exe /f'
                os.system(os_command)
                # Create an empty list to store conversation history.
                conversation = list()
                # Define Chrome options.
                options = Options()
                options.add_experimental_option("excludeSwitches", ["enable-logging"])
                # Add a user data directory as an argument for options.
                options.add_argument(f"--user-data-dir=C:\\Users\\{user}\\AppData\\Local\\Google\\Chrome\\User Data")
                options.add_argument("profile-directory=Default")
                # In order to prevent the "Timed out receiving message from renderer: 20.000" error, add the following options.
                options.add_experimental_option('extensionLoadTimeout', 60000)
                # Instansiate Google Chrome with the above options. 
                driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
                selenium_client = Selenium
                selenium_client.setup(driver)
                # Pause for 2 seconds.
                time.sleep(2)
                # Create an infinite loop.
                while True:
                    selenium_client.open_quora_questions(driver)
                    # Pause for 2 seconds.
                    time.sleep(2)
                    question = selenium_client.extract_question(driver)
                    answer = selenium_client.gpt3_completion(question)
                    selenium_client.answer_quora_question(driver, answer)
                    #selenium_client.quit()
        # If it isn't, then ask the user how they want to enter their OpenAI API key.
        else:
            print("OpenAI API key not set in environment.")
            # Get the OpenAI API key from the user.
            api_key = set_openai_api_key.get_openai_api_key()
            # Set the OpenAI API key in the environment.
            set_openai_api_key.set_openai_api_key(api_key)
            # Open the environment variables in the Windows GUI with OpenAI API key highlighted.
            set_openai_api_key.open_env()
            # Run the script.
         

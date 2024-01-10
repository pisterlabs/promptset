import time
from selenium.webdriver.common.by import By
#import ActionChains and WebDriverWait and EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import requests
import datetime
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import openai
from selenium import webdriver
from auto_influencer.prompts import prompts



def use_prompt_for_img_gen(driver):
    gen_button = driver.find_element(By.CSS_SELECTOR, '#generate_button')
    prompt_box = driver.find_element(By.CSS_SELECTOR, '#positive_prompt > label > textarea')

    for prompt in prompts:
        prompt_box.send_keys(prompt)
        gen_button.click()
        time.sleep(600)  # Wait for 10 minutes
        prompt_box.clear()
        time.sleep(5)

def use_img_gen(driver,reference_image1,reference_image2,reference_image3,reference_image4, optional_prompt, negative_prompt, performance, style, number_of_images, aspect_Ratio):
    #toggle input image
    input_image_checkbox = driver.find_element(By.CSS_SELECTOR, '#component-16 > label > input')
    input_image_checkbox.click()
    time.sleep(.5)
    #click advanced
    advanced_button = driver.find_element(By.CSS_SELECTOR, '#component-62 > label > input')
    advanced_button.click()
    time.sleep(.5)
    if reference_image1 != None:
        #upload image 1
        image1 = driver.find_element(By.CSS_SELECTOR, '#component-18 > div > div > div > div > div > div > div > div > div > div > div > div > div > div > input[type=file]')
        image1.send_keys(reference_image1)
        time.sleep(.5)

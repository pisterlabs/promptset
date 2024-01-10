import logging
from os import environ as env
import time
import openai
import telebot
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import markdownify
import undetected_chromedriver as uc
from fake_useragent import UserAgent
ua = UserAgent()

options = webdriver.ChromeOptions()
options.headless = False
user_agent = ua.random
options.add_argument(f'user-agent={user_agent}')


logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)

bot = telebot.TeleBot(env["TELEGRAM_BOT_KEY"])
openai.api_key = env["OPENAI_API_KEY"]


@bot.message_handler(func=lambda message: True)
def get_chatgpt(message):
    driver = uc.Chrome(options=options)
    driver.get("https://chat.openai.com")
    
    elem = WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.XPATH, '//button[text()="Log in"]')) 
    )
    driver.find_element(By.XPATH, '//button[text()="Log in"]').click()
    elem = WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.ID, "username")) 
    )

    driver.get(driver.current_url)
    username = driver.find_element(By.ID, "username")
    username.send_keys(env["OPENAI_USERNAME"])

    # find iframe
    captcha_iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (
                By.TAG_NAME, 'iframe'
            )
        )
    )

    ActionChains(driver).move_to_element(captcha_iframe).click().perform()

    # click im not robot
    captcha_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (
                By.ID, 'g-recaptcha-response'
            )
        )
    )

    driver.execute_script("arguments[0].click()", captcha_box)

    time.sleep(2)

    driver.find_element(By.XPATH, '//button[text()="Continue"]').click()
    password = driver.find_element(By.ID, "password")
    password.send_keys(env["OPENAI_PASSWORD"])
    driver.find_element(By.XPATH, '//button[text()="Continue"]').click()

    WebDriverWait(driver, 60).until(
    EC.presence_of_element_located((By.XPATH, '//button[text()="Next"]')) 
    )

    driver.find_element(By.XPATH, '//button[text()="Next"]').click()
    driver.find_element(By.XPATH, '//button[text()="Next"]').click()
    driver.find_element(By.XPATH, '//button[text()="Done"]').click()
    input_prompt = driver.find_element(By.CSS_SELECTOR, "textarea.w-full")
    input_prompt.send_keys(message.text)
    input_prompt.send_keys(Keys.ENTER)

    WebDriverWait(driver, 90).until(
    EC.presence_of_element_located((By.XPATH, "//button[contains(text(), 'Try again')]")) 
    )
    response=driver.find_element(By.XPATH, "//div[contains(@class, 'markdown prose')]").get_attribute('innerHTML')
    response = markdownify.markdownify(response, heading_style="ATX")
    print(response)

    bot.send_message(message.chat.id,
                     f'{response}',
                     parse_mode="markdown")
    driver.close()

bot.infinity_polling(timeout=120, long_polling_timeout=240)

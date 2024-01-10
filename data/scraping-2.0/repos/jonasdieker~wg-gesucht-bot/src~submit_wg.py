# -*- coding: utf-8 -*-
import json
import time

from selenium import webdriver
from selenium.common.exceptions import (
    ElementNotInteractableException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src import OpenAIHelper


def get_element(driver, by, id):
    ignored_exceptions = (
        StaleElementReferenceException,
        NoSuchElementException,
        ElementNotInteractableException,
    )
    remove_cookies_popup(driver)
    try:
        wait = WebDriverWait(driver, 10, poll_frequency=1)
        element = wait.until(EC.visibility_of_element_located((by, id)))
    except TimeoutException:
        wait = WebDriverWait(driver, 30, poll_frequency=1)
        element = wait.until(EC.presence_of_element_located((by, id)))
    if isinstance(element, list):
        element = element[0]
    return element


def click_button(driver, by, id):
    remove_cookies_popup(driver)
    driver.implicitly_wait(2)
    try:
        element = get_element(driver, by, id)
        driver.implicitly_wait(2)
        element.click()
    except ElementNotInteractableException:
        raise ElementNotInteractableException()


def click_at_coordinates(driver, x, y):
    remove_cookies_popup(driver)
    driver.implicitly_wait(2)
    action = ActionChains(driver)
    action.move_by_offset(x, y).click().perform()


def remove_cookies_popup(driver):
    divs_to_remove = driver.find_elements(
        By.XPATH, "//div[@id='cmpbox' or @id='cmpbox2']"
    )
    for div in divs_to_remove:
        driver.execute_script("arguments[0].remove();", div)


def send_keys(driver, by, id, send_str):
    driver.implicitly_wait(2)
    try:
        element = get_element(driver, by, id)
        element.send_keys(send_str)
    except ElementNotInteractableException:
        raise ElementNotInteractableException(f"Could not enter: {send_str}")


def gpt_get_language(config, logger) -> str:
    openai = OpenAIHelper(config["openai_credentials"]["api_key"])
    listing_text = config["listing_text"]

    if len(listing_text) < 200:
        listing_text = listing_text[10:]
    else:
        listing_text = listing_text[10:200]

    # build prompt
    prompt_lst = []
    prompt_lst.append("What language is this:\n")
    prompt_lst.append(listing_text)
    prompt_lst.append(" Please only respond in a JSON style format like ")
    prompt_lst.append('{"language": "<your-answer>"}, '),
    prompt_lst.append(
        "where your answer should be a single word which is the language."
    )
    prompt = "".join(prompt_lst)

    # pass prompt into openai helper method
    response = str(openai.generate(prompt)).strip()
    # response = '{"language": "German"}'
    logger.info(f"GPT response: {response}")

    # try to load json from string
    try:
        response_json = json.loads(response)
    except ValueError:
        logger.info("Response was not valid JSON format.")
        return ""
    return response_json.get("language", "")


def gpt_get_keyword(config, logger) -> str:
    openai = OpenAIHelper(config["openai_credentials"]["api_key"])
    listing_text = config["listing_text"]

    prompt_lst = []
    prompt_lst.append(
        "Check if there exists a keyword in the text, to show I read the text.\n"
    )
    prompt_lst.append(
        'This keyword is most likely wrapped in quotation marks like: "".\n'
    )
    prompt_lst.append(
        "Note however that not all texts will include such a keyword. Here is the text:\n"
    )
    prompt_lst.append(f"'{listing_text}'")
    prompt_lst.append("\nPlease only respond in JSON format like ")
    prompt_lst.append('{"keyword": "<your-keyword>"}, ')
    prompt_lst.append(
        "where your <your-keyword> is the keyword from the text that you found."
    )
    prompt = "".join(prompt_lst)

    response = str(openai.generate(prompt)).strip()
    logger.info(f"GPT response: {response}")
    try:
        response_json = json.loads(response)
    except ValueError:
        logger.info("Response was not valid JSON format.")
        return ""
    return response_json.get("keyword", "")


def submit_app(config, logger):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--log-level=3")

    # add the argument to reuse an existing tab
    if config["run_headless"]:
        chrome_options.headless = True
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--reuse-tab")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # create the ChromeDriver object and log
    try:
        service_log_path = "chromedriver.log"
        service_args = ["--verbose"]
        driver = webdriver.Chrome(
            config["chromedriver_path"],
            options=chrome_options,
            service_args=service_args,
            service_log_path=service_log_path,
        )
        # mainly when using screen
        driver.maximize_window()
        driver.get("https://www.wg-gesucht.de/nachricht-senden" + config["ref"])
    except Exception as e:
        logger.log(
            "Chrome crashed! You might be trying to run it without a screen in terminal?"
        )
        raise e

    # # accept cookies button
    # click_button(driver, By.XPATH, "//*[contains(text(), 'Accept all')]")
    # click_button(driver, By.XPATH, "//*[contains(text(), 'Akzeptieren')]")
    # instead of accepting cookies, just remove cookie popup,
    # as "Akzeptieren" doesn't show on headless Firefox on Linux for some reason
    remove_cookies_popup(driver)

    # my account button
    click_button(driver, By.XPATH, "//*[contains(text(), 'Mein Konto')]")

    # enter email
    send_keys(
        driver, By.ID, "login_email_username", config["wg_gesucht_credentials"]["email"]
    )

    # enter password
    send_keys(
        driver, By.ID, "login_password", config["wg_gesucht_credentials"]["password"]
    )

    # login button
    click_button(driver, By.ID, "login_submit")
    logger.info("Logged in.")

    remove_cookies_popup(driver)

    # remove lightbox div that blocks attachments
    divs_to_remove = driver.find_elements(By.XPATH, "//div[@class='lightbox']")
    for div in divs_to_remove:
        driver.execute_script("arguments[0].remove();", div)

    # occasionally wg-gesucht gives you advice on how to stay safe.
    try:
        click_button(driver, By.ID, "sicherheit_bestaetigung")
    except:
        logger.info("No security check.")

    driver.implicitly_wait(2)

    # checks if its possible to sent message to listing.
    try:
        _ = get_element(driver, By.ID, "message_timestamp")
        logger.info("Message has already been sent previously. Will skip this offer.")
        driver.quit()
        return False
    except:
        logger.info("No message has been sent. Will send now...")

    logger.info(f"Sending to: {config['user_name']}, {config['address']}.")

    driver.implicitly_wait(2)

    text_area = get_element(driver, By.ID, "message_input")
    if text_area:
        text_area.clear()

    # get language from GPT
    languages = list(config["messages"].keys())
    if len(languages) == 1:
        message_file = config["messages"][languages[0]]
        logger.info(f"Selected text in {languages[0]}")
    else:
        if config["openai_credentials"]["api_key"] == "":
            # no openai key so just use first language in list
            message_file = config["messages"][languages[0]]
            logger.info(f"No openai api key -> selected text in {languages[0]}")
        else:
            # check which languages user want to message in using GPT
            listing_language = gpt_get_language(config, logger).lower()
            if listing_language in languages:
                message_file = config["messages"][listing_language]
                logger.info(f"Selected text in {listing_language}.")
            else:
                logger.info(
                    f"Listing language does not overlap with available texts in {languages}. Using first languages in list."
                )
                message_file = config["messages"][languages[0]]

    # if config["openai_credentials"]["api_key"] != "":
    #     # get keyword from GPT
    #     keyword = gpt_get_keyword(config, logger)

    # read message from a file
    try:
        with open(message_file, "r") as file:
            message = str(file.read())
        message = message.replace("receipient", config["user_name"].split(" ")[0])
        # if "keyword" in locals() and keyword != "":
        #     message = f"{keyword}\n\n" + message
        text_area.send_keys(message)
    except:
        logger.info(f"{message_file} file not found!")
        driver.quit()
        return False

    driver.implicitly_wait(2)

    try:
        click_button(
            driver,
            By.XPATH,
            "//button[@data-ng-click='submit()' or contains(.,'Nachricht senden')]",
        )
        logger.info(f">>>> Message sent to: {config['ref']} <<<<")
        time.sleep(2)
        driver.quit()
        return True
    except ElementNotInteractableException:
        logger.info("Cannot find submit button!")
        driver.quit()
        return False

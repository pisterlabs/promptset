from dotenv import load_dotenv
import openai
import os
import config
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import NoSuchWindowException
import time
import sys

TEXTAREA_MAX_TIMEOUT = 99999


def fetch_textarea(driver):
    return WebDriverWait(driver, TEXTAREA_MAX_TIMEOUT).until(
        EC.visibility_of_element_located(
            (
                By.CSS_SELECTOR,
                "textarea.block.w-full.resize-none.overflow-y-hidden.whitespace-pre-wrap.bg-transparent.outline-none.placeholder\\:text-brand-gray-400.font-serif.font-normal.text-body-chat-m.lg\\:text-body-chat-l",
            )
        )
    )


def interpret(transcript_queue):
    print("Starting browser...")
    driver = webdriver.Chrome()
    driver.get("https://pi.ai/talk")

    PAGE_LOADING_WAIT_TIME = 4
    print("Waiting {}s for page to load...".format(PAGE_LOADING_WAIT_TIME))
    time.sleep(PAGE_LOADING_WAIT_TIME)

    print("Waiting for textarea...")
    textarea = fetch_textarea(driver)

    print("Interpreting...")
    while True:
        # Get the transcript from the queue
        transcript = transcript_queue.get()

        # If we receive exit code, stop
        if transcript == config.EXIT_CODE:
            break

        # Load environment variables from .env
        load_dotenv()

        openai.api_key = os.getenv("OPENAI_API_KEY")

        # MAX_TOKENS = 32

        # completion = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "Reply Y if an AI should reply to this, otherwise N",
        #         },
        #         {"role": "user", "content": transcript},
        #     ],
        #     max_tokens=MAX_TOKENS,
        # )

        # # Trim and clean the message choice
        # is_question = completion.choices[0].message.content.strip()

        # print(f'IS_QUESTION: "{is_question}"')

        # If it is a question, ask pi ai
        # if is_question == "Y" or is_question == "y":
        RETRY_DELAY = 2
        retry = False
        while True:
            try:
                if retry == True:
                    print("Retrying in {}s...".format(RETRY_DELAY))
                    time.sleep(RETRY_DELAY)

                retry = True
                textarea.clear()
                textarea.send_keys(transcript)
                time.sleep(0.2)
                textarea.send_keys(Keys.RETURN)
                break
            except StaleElementReferenceException:
                print("Element is not attached to the page document")
                textarea = fetch_textarea(driver)
            except NoSuchElementException:
                print("Element does not exist anymore on the DOM")
                textarea = fetch_textarea(driver)
            except NoSuchWindowException:
                print("Restart the app. Tabs were messed up with the handle is lost")
            except:
                print("Unexpected error:", sys.exc_info()[0])

    print("Closing browser...")
    del textarea
    driver.quit()

    print("Stopped interpreting")

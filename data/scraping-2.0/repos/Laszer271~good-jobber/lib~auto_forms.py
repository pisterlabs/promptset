import json
import os
import asyncio
from time import sleep
from bs4 import BeautifulSoup
from lxml import etree

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.tools.playwright.utils import (
    create_async_playwright_browser,
    create_sync_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.
)
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, AgentType
# from playwright.async_api import async_playwright
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By

from lxml import etree
from io import StringIO


def read_data():
    with open("../data/example_personal_info.json") as json_file:
        data_personal = json.load(json_file)

    with open("../data/job_offers.json") as job_file:
        data_jobs = json.load(job_file)

    return data_jobs, data_personal


def get_element_by_xpath(driver, xpath, timeout=10, time_wait=0.5):
    sleep(time_wait)
    # This could print element's text:
    # source = driver.page_source
    # htmlparser = etree.HTMLParser()
    # tree = etree.parse(StringIO(source), htmlparser)
    # print(tree.xpath(xpath)[0].text)

    return WebDriverWait(driver, timeout).until(lambda x: x.find_element('xpath', xpath))


def click_element_by_xpath(driver, xpath, **kwargs):
    element = get_element_by_xpath(driver, xpath, **kwargs)
    element.click()
    return element


def fill_element_by_xpath(driver, xpath, text, **kwargs):
    element = get_element_by_xpath(driver, xpath, **kwargs)
    element.send_keys(text)
    return element


def run_selenium(url, personal_data):
    driver = webdriver.Firefox()
    driver.get(url)

    # Click "Apply" button
    xpathButtonApply = "//*[@id='applyButton']"
    click_element_by_xpath(driver, xpathButtonApply)

    # Fill in the "Name & Surname"
    xpathNameInput = '//*[@id="apply-modal"]/section/common-material-modal/div/section/nfj-apply-internal-step-application/form/nfj-form-field[1]/div[1]/div/input'
    fill_element_by_xpath(driver, xpathNameInput, f"{personal_data['Name']} {personal_data['Surname']}")

    # Fill in the "Email"
    xpathEmailInput = '//*[@id="apply-modal"]/section/common-material-modal/div/section/nfj-apply-internal-step-application/form/nfj-form-field[2]/div[1]/div/input'
    fill_element_by_xpath(driver, xpathEmailInput, personal_data['Email'])

    # Fill in the "job location" (in 3 steps: click, choose, close selection)
    # Click "Choose job location"
    xpathChooseJob = '//*[@id="apply-modal"]/section/common-material-modal/div/section/nfj-apply-internal-step-application/form/div[1]/div/nfj-multiselect-dropdown/div/div/div[1]/span'
    click_element_by_xpath(driver, xpathChooseJob)

    # Tutaj trzeba użyć LLM żeby znaleźć lokalizacje najbardziej odpowiadajace użytkownikowi
    # Select location that is best for candidate
    xpathLocationCheckbox = '//*[@id="apply-modal"]/section/common-material-modal/div/section/nfj-apply-internal-step-application/form/div[1]/div/nfj-multiselect-dropdown/div/div/div[2]/div/ul[2]/li[1]'
    click_element_by_xpath(driver, xpathLocationCheckbox)

    # Close selection
    click_element_by_xpath(driver, xpathChooseJob)

    # Upload CV 
    xpathCVUpload = '//*[@id="attachment"]'
    cv_path = personal_data['cv_path']
    fill_element_by_xpath(driver, xpathCVUpload, cv_path)

    # Probably not needed during demo? Is it even needed at all? 
    # Seems like the bot can click stuff even if it's not visible because of the popup

    # xpathCookies = '//*[@id="onetrust-accept-btn-handler"]'
    # WebDriverWait(driver, 30).until(lambda x: x.find_element('xpath', xpathCookies))
    # acceptCookies = driver.find_element('xpath', xpathCookies)
    # acceptCookies.click()
    
    sleep(4)
    driver.close()


def findBestLocation(htmlCode, locationUser):
    pass


if __name__ == '__main__':
    data_jobs, data_personal = read_data()
    url = data_jobs[8]["Url"]
    run_selenium(url, data_personal)
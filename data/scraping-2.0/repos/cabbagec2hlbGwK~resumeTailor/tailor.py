#!/bin/python3
import openai
from selenium import webdriver
from bs4 import BeautifulSoup
import os
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


def main():
    openAI()


def getJobPosting(url):
    fireFoxOptions = webdriver.FirefoxOptions()
    # fireFoxOptions.headless = True
    driverPath = GeckoDriverManager().install()
    brower = webdriver.Firefox(
        options=fireFoxOptions,
        service=Service(executable_path=driverPath),
    )
    brower.get(url)
    r = brower.page_source
    brower.quit()
    return r


def indeedHandeler(url):
    rawData = getJobPosting(url)
    data = {}
    soup = BeautifulSoup(rawData, "html5lib")
    data["jobTitle"] = (
        soup.find("h1", {"class": "jobsearch-JobInfoHeader-title"}).find("span").text
    )
    data["company"] = (
        soup.find("div", {"data-testid": "inlineHeader-companyName"}).find("a").text
    )
    data["jobDiscription"] = soup.find("div", {"id": "jobDescriptionText"}).text
    return data


def linkedinHandeler(url):
    rawData = getJobPosting(url)
    data = {}
    soup = BeautifulSoup(rawData, "html5lib")
    data["jobTitle"] = soup.find("h1", {"class": "top-card-layout__title"}).text
    data["company"] = soup.find("a", {"class": "topcard__org-name-link"}).text.strip()
    data["jobDiscription"] = soup.find(
        "div", {"class": "show-more-less-html__markup"}
    ).text
    return data


def jobBoardHandeler(url):
    match url:
        case url if "indeed" in url:
            indeedHandeler(url)
        case url if "linkedin" in url:
            linkedinHandeler(url)
        case _:
            print("job board is not supported")


def openAI():
    openai.organization = "org-3xUaVEFCSYFk0luVN814ZrvW"
    openai.api_key = os.getenv("gpt")
    print(openai.Model.list())


if "__main__" == __name__:
    main()

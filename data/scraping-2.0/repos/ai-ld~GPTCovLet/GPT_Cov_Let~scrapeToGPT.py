import openai
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# pytest main.py --cov


def urlScrape(url):
    options = Options()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    username = driver.find_element(By.ID, "username")
    username.send_keys("jaesungpark271@gmail.com")
    pword = driver.find_element(By.ID, "password")
    pword.send_keys("")
    driver.find_element(By.XPATH, "//button[@type='submit']").click()

    driver.get(url)
    time.sleep(3)
    src = driver.page_source
    html = BeautifulSoup(src, "html.parser")

    return descScrape(html)


def descScrape(html):
    # print(html.prettify())
    company_name_html = html.find_all("a", {"class": "ember-view t-black t-normal"})
    # company_name_html = html.find_all('div', {'class': 'jobs-unified-top-card__primary-description'})
    company_name = (
        company_name_html[0].text
    ).strip()  # if there is an error here it means you need to input your linkedin user email and password in the urlScrape Function

    # print(company_name)
    return company_name


# def pull templateCoverLetter or Prompt():


def completionQuery(desc):
    openai.api_key = ""

    # pull templateCoverLetterHere
    # cap completion to 10tokens
    prompt = (
        "Write a three paragraph cover letter for the position of software developer to "
        + desc
        + " as a soon to be graduate in major of computer science at Columbia University."
    )
    print("Would be Prompt: Costs Tokens")
    print(prompt)
    # completion = openai.Completion.create(model="text-davinci-003", prompt, max_tokens = 750)
    # print(completion.choices[0].text)
    return True


url = "https://www.linkedin.com/jobs/view/dev10-entry-level-software-developer-nationwide-at-dev10-3497504875/?utm_campaign=google_jobs_apply&utm_source=google_jobs_apply&utm_medium=organic"
# urlScrape(url)

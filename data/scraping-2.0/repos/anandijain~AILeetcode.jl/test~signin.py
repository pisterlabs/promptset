from time import sleep
import json
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import openai
# initialize the driver
langs = ["c++", "java", "python3", "c", "c#", "javascript", "ruby", "swift", "go", "scala", "kotlin", "rust", "php", "typescript", "racket", "erlang", "elixir", "dart"]
driver = webdriver.Firefox()
def login(driver):
    driver.get("https://leetcode.com/accounts/login/")
    wait = WebDriverWait(driver, 10)
    sleep(5)
    email = wait.until(EC.presence_of_element_located((By.NAME, "login")))
    email.send_keys("anandj@uchicago.edu")
    password = wait.until(EC.presence_of_element_located((By.NAME, "password")))
    password.send_keys("$(5GwewCf%~QRLN")
    password.send_keys(Keys.RETURN)
    sleep(5)
    wait = WebDriverWait(driver, 10)


login(driver)

# Wait for the page to load and verify login success
assert "LeetCode" in driver.title
df = pd.read_csv('probs.csv')

langs = ['c++', 'python3', 'typescript', 'rust']
btns = driver.find_elements('tag name', 'button')

def get_btns(langs):
    return [btn for btn in btns if btn.text.lower() in langs]


problem_texts = []
rows = list(df.iterrows())
for index, row in rows:
    print(row)
    link = row['prob_url']
    driver.get(link)
    wait = WebDriverWait(driver, 10)
    sleep(5)
    xx = driver.find_element('xpath', "/html/head/meta[@name='description']")
    c = xx.get_attribute("content")
    # filename = f"data/{row['prob_id']}.txt"
    # with open(filename, 'w') as f:
    #     f.write(c)
    filepath = f'/Users/anand/.julia/dev/AILeetcode/data/starter/{row.prob_id}.json'

    if os.path.exists(filepath):
        with open(filepath) as f:
            d = json.load(f)
    else:
        d = {}
    for lang in langs:
        try:
            btns = driver.find_elements('tag name', 'button')
            btns = get_btns(langs)
            btn = btns[0]
        except:
            continue
        try:
            # print(lang)
            btn.click()
            langbtns = btn.find_elements("xpath", "//li")
            curr_langbtns = [btn for btn in langbtns if btn.text.lower() == lang]

            if len(curr_langbtns) > 0:
                curr_langbtns[0].click()
            else:
                continue
            sleep(0.2)

            line_es = driver.find_elements('css selector', "div.view-line")
            # print(list(map(lambda x: x.text, line_es)))
            starter_code = '\n'.join([line.text for line in line_es])
            d[lang] = starter_code
        except:
            d = {}
            btns = driver.find_elements('tag name', 'button')
            btns = get_btns(langs)
            btn = btns[0]
            # print(lang)
            btn = get_btn(langs)
            btn.click()
            langbtns = btn.find_elements("xpath", "//li")
            # print(list(map(lambda x: x.text, langbtns)))
            lb = only([btn for btn in langbtns if btn.text.lower() == lang])
            lb.click()

            # print()
            if len(curr_langbtns) > 0:
            # else:
            #     return None
            sleep(0.2)
            line_es = driver.find_elements('css selector', "div.view-line")
            starter_code = '\n'.join([line.text for line in line_es])
            # print(starter_code)
            d[lang] = starter_code
    print(d)
    folder_path = '/Users/anand/.julia/dev/AILeetcode/data/starter'
    filename = f'{row.prob_id}.json'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filepath = os.path.join(folder_path, filename)

    with open(filepath, 'w') as f:
        json.dump(d, f)

    print(f'Starter code written to {filepath}')
   
d = {}
# def get_starter_code(driver, lang):
for lang in langs:
    btns = driver.find_elements('tag name', 'button')
    btns = get_btns(langs)
    btn = btns[0]
    print(lang)
    btn.click()
    langbtns = btn.find_elements("xpath", "//li")
    print(list(map(lambda x: x.text, langbtns)))
    curr_langbtns = [btn for btn in langbtns if btn.text.lower() == lang]

    # print()
    if len(curr_langbtns) > 0:
        curr_langbtns[0].click()
    # else:
    #     return None

    line_es = driver.find_elements('css selector', "div.view-line")
    starter_code = '\n'.join([line.text for line in line_es])
    print(starter_code)
    d[lang] = starter_code
# return starter_code

# for lang in langs:
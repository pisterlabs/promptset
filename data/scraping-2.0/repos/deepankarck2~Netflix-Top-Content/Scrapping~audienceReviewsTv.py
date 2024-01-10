import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from sqlalchemy import create_engine
import time
from selenium.webdriver.common.keys import Keys
import urllib.request as request
import openai
import re
from string import punctuation

def rt_url(title):
    name = r'.+(?=:)'
    na = re.search(name,title)
    new = na.group(0).translate(str.maketrans('', '', punctuation))
    l = re.sub('\s','_',new).lower()

    
    num = r'\d+$'
    n = re.search(num,title)
    if n ==None:
        return 'https://www.rottentomatoes.com/tv/'+ l

    
    nu = n.group(0)
    link ='https://www.rottentomatoes.com/tv/'+ l +'/s0' + nu
    return link
def scrape(show):
    list = []
    rotten = rt_url(show)
    print(rotten)
    driver.get(rotten)
    time.sleep(5) 
    try:
        driver.find_element(By.XPATH,'//*[@id="scoreboard"]/a[2]').click()
    except:
        driver.find_element(By.XPATH,'//*[@id="seasons-list"]/div/a/season-list-item').click()
        driver.find_element(By.XPATH,'//*[@id="scoreboard"]/a[2]').click()
        time.sleep(3)
    
    reviews = driver.find_elements(By.CSS_SELECTOR,'.audience-reviews__review.js-review-text')
    rev = reviews
    for r in rev:
        res = len(re.findall(r'\w+', r.text))
        if res >= 80:
            list.append(r.text)
    time.sleep(8)
    try:
        next = driver.find_element(By.XPATH,'//*[@id="reviews"]/div[1]/rt-button[2]')
        next.click()
    except:
        print(len(list))
        #df[show] = pd.Series(list)
        time.sleep(5)
        return list
    time.sleep(8)
    while len(list) <= 40:

        reviews = driver.find_elements(By.CSS_SELECTOR,'.audience-reviews__review.js-review-text')
        rev = reviews
        for r in rev:
            res = len(re.findall(r'\w+', r.text))
            if res >= 80:
                list.append(r.text)
        time.sleep(7)
        try:
            next = driver.find_element(By.XPATH,'//*[@id="reviews"]/div[1]/rt-button[2]')
            next.click()
            time.sleep(8)
        except:
            break
    print(len(list))
    #df[show] = pd.Series(list)
    time.sleep(5)
    return list



path = "chromedriver.exe"

engine = create_engine("mysql://admin:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/netflix")
test = pd.read_sql_query('select TV from netflixTopTv10',engine)
tv = test['TV'].to_list()

driver = webdriver.Chrome(path) 


df = pd.DataFrame()
for show in tv:
    try:
        df[show] = pd.Series(scrape(show))

    except:
        driver.quit()
        df[show] = pd.Series(scrape(show))
        continue
print(df)
df.to_sql('audienceReviewsTv', con=engine, if_exists='replace')
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from sqlalchemy import create_engine
import time
from selenium.webdriver.common.keys import Keys
import urllib.request as request
import openai
import re
from string import punctuation
from selenium.webdriver.support import expected_conditions as EC

def rt_url(title):
    new = title.translate(str.maketrans('', '', punctuation))
    l = re.sub('\s','_',new).lower()
    link  = 'https://www.rottentomatoes.com/m/'+ l
    return link
def scrape(movie):
    list = []
    rotten = rt_url(movie)
    print(rotten)
    driver.get(rotten)
    time.sleep(5) 
    try:
        driver.find_element(By.XPATH,'//*[@id="scoreboard"]/a[2]').click()
        time.sleep(4)
        driver.find_element(By.XPATH,'//*[@id="reviews"]/nav/ul/li[3]').click()
    except:
        driver.get("http://www.google.com")
        google = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, "q")))
        google.send_keys(movie + " rotten tomatoes")
        time.sleep(5)
        google.send_keys(Keys.ENTER)
        result = driver.find_element(By.CSS_SELECTOR,'.LC20lb.MBeuO.DKV0Md')
        result.click()    
        driver.find_element(By.XPATH,'//*[@id="scoreboard"]/a[2]').click()   
        time.sleep(3)
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
    while len(list) <= 15:

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
driver = webdriver.Chrome(path) 
engine = create_engine("mysql://admin:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/netflix")
test = pd.read_sql_query('select Movie from netflixTopMovie10',engine)
Movie = test['Movie'].to_list()

df = pd.DataFrame()
for m in Movie:
    try:
        df[m] = pd.Series(scrape(m))

    except:
        driver.quit()
        time.sleep(3)
        df[m] = pd.Series(scrape(m))
        continue
print(df)
df.to_sql('audienceReviewsMovie', con=engine, if_exists='replace')
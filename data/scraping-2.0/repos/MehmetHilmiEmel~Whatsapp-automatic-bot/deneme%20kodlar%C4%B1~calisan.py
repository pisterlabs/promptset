from selenium import webdriver
import requests
from bs4 import BeautifulSoup as bs
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import openai
import time
import random




def start():
    driver=webdriver.Chrome()
    driver.implicitly_wait(3)
    driver.get("https://web.whatsapp.com/")
    openai.api_key = "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"
    while True:
        try:
            qr_area=driver.find_element(By.XPATH,'//*[@id="app"]/div/div/div[3]/div[1]/div/div/div[2]/div')
            time.sleep(2)
        except:
            message=driver.find_elements(By.XPATH," //span[@data-testid='icon-unread-count']")
            my=driver.find_elements(By.XPATH," //span[@data-testid='you-label']")
            for i in message:
                i.click()
                time.sleep(1)
                about=driver.find_elements(By.XPATH,"//*[@id='main']/header/div[2]/div[1]/div/span")
                about[0].click()
                time.sleep(2)
                wp_source=driver.page_source
                soup=bs(wp_source,"html.parser")
                grup=soup.find('span',{'class':['enbbiyaj','e1gr2w1z','hp667wtd']})
                print(grup.text)
                if "Grup" in grup.text:
                    time.sleep(2)
                    my[0].click()
                    time.sleep(2)
                else:
                    close=driver.find_element(By.XPATH,"//*[@id='app']/div/div/div[6]/span/div/span/div/header/div/div[1]/div/span")
                    close.click()
                    time.sleep(2)
                   
                    name=driver.find_element(By.XPATH," //span[@data-testid='conversation-info-header-chat-title']")
                    print(name.text)
                    wp_source=driver.page_source
                    soup=bs(wp_source,"html.parser")
                    time.sleep(2)
                    okunmamis=soup.find_all('span',{'class':'_2jRew'})
                    print(okunmamis[-1].text)
                    if "okunmamış" in okunmamis[-1].text:
                        time.sleep(2)
                        
                        wp_source=driver.page_source
                        soup_end=bs(wp_source,"html.parser")
                        search=soup_end.find_all('span',{'class':['_11JPr','selectable-text', 'copyable-text']})
                        
                        son_tweet=search[-1].find_all('span')[0].text
                        text=son_tweet

                        print(text)
                        while True:
                            try:
                                messages = [
                                {"role": "system", "content": text},
                                ]

                                model = "gpt-3.5-turbo-0301"
                                response_open = openai.ChatCompletion.create(
                                model=model,
                                messages=messages,
                                temperature=0,
                                )

                                # Start conversation with chatbot
                                text_gpt=response_open["choices"][0]["message"]["content"]
                                
                                print(text_gpt)
                                time.sleep(1)
                                message_area=driver.find_element(By.XPATH,"//*[@id='main']/footer/div[1]/div/span[2]/div/div[2]/div[1]/div/div[1]/p")
                                message_area.click()
                                message_area.send_keys(text_gpt)
                                message_area.send_keys(Keys.ENTER)
                                break
                            except:
                                time.sleep(1)
                        time.sleep(2)
                        my[0].click()
                        
                        

                    else:
                
                        wp_source=driver.page_source
                        soup=bs(wp_source,"html.parser")
                        search=soup_end.find_all('span',{'class':['_11JPr','selectable-text', 'copyable-text']})
                        print(search)
                        son_tweet=search[-1].find_all('span')[0].text
                        text=son_tweet

                        print(text)
                

                print("you entered qr code")
                time.sleep(2)
    
start()
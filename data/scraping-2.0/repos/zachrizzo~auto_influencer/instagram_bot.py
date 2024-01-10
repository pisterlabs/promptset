from selenium.webdriver.common.by import By
#import ActionChains and WebDriverWait and EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import requests
import datetime
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import openai
from selenium import webdriver

# service = ChromeService(executable_path=ChromeDriverManager().install())



class InstaBot():
    def __init__(self):
        self.driver = None





    def open_insta(self, fullScreen: bool = False):
        # if not self.driver:
            self.driver = webdriver.Firefox()
            self.driver.get('https://www.instagram.com')
            if fullScreen:
                self.driver.maximize_window()
            time.sleep(5)




    def login(self, email, password):
        time.sleep(3)
        email_input = self.driver.find_element(By.CSS_SELECTOR, '#loginForm > div > div:nth-child(1) > div > label > input')
        email_input.send_keys(email)
        password_input = self.driver.find_element(By.CSS_SELECTOR, '#loginForm > div > div:nth-child(2) > div > label > input')
        password_input.send_keys(password)
        time.sleep(1)
        login_button = self.driver.find_element(By.CSS_SELECTOR, '#loginForm > div > div:nth-child(3) > button')
        login_button.click()
        time.sleep(5)
        return True


    def _search(self):
        search_icon = self.driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div/div/div[2]/div[2]/span')
        search_icon.click()

    def copy_images(self, accounts_to_copy, path_to_save_images):
        if self.driver == None:
            print('driver not initialized')
            return
        time.sleep(2)
        self._search()
        time.sleep(2)

        for account in accounts_to_copy:
            search_bar = self.driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div[2]/div/div/div[2]/div/div/div[1]/div/div/input')
            search_bar.clear()
            search_bar.send_keys(account)
            time.sleep(3)
            first_option = self.driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div[2]/div/div/div[2]/div/div/div[3]/div/a[1]/div[1]/div/div/div[2]/div/div')
            first_option.click()
            time.sleep(6)
            while True:
                    try:

                        insta_user = self.driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/div[2]/section/main/div/header/section/div[1]/a/h2')

                        if(insta_user.text == account):
                            print('page loaded')
                            break
                        else:
                            time.sleep(1)
                            print('waiting for page to load')
                            continue
                    except:
                        time.sleep(1)
                        continue

                # body = driver.find_element(By.XPATH, '/html/body')
                # body.click()

            search_icon = self.driver.find_element(By.XPATH, '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div/div/div[2]/div[2]/span')
            search_icon.click()



            try:
                last_row_index = 0
                while True:
                    rows_of_posts = self.driver.find_elements(By.XPATH, '/html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/div[2]/section/main/div/div[3]/article/div[1]/div/div')
                    if len(rows_of_posts) <= last_row_index:
                        break  # Break the loop if no new rows are found
                    print(f'Found {len(rows_of_posts) - last_row_index} new rows of posts')
                    for row_index in range(last_row_index, len(rows_of_posts)):
                        row = rows_of_posts[row_index]
                        self.driver.execute_script("arguments[0].scrollIntoView();", row)
                        time.sleep(1)
                        # /html/body/div[2]/div/div/div[2]/div/div/div[1]/div[1]/div[2]/div[2]/section/main/div/div[2]/article/div[1]/div
                        posts_in_row = row.find_elements(By.CSS_SELECTOR, 'div > div > a ')
                        for post_index, post in enumerate(posts_in_row):
                            print('scrolling to post')
                            self.driver.execute_script("arguments[0].scrollIntoView();", post)
                            time.sleep(1)

                            try:
                                # Attempt to click the target element
                                img = post.find_element(By.CSS_SELECTOR, "div")
                                img.click()
                                time.sleep(2)
                            except Exception as e:
                                print("Encountered an error, trying to click the blocking element:", e)

                                # Find and click the blocking element
                                blocking_element = post.find_element(By.TAG_NAME, "div")
                                blocking_element.click()
                                time.sleep(1)


                            large_photo = self.driver.find_element(By.CSS_SELECTOR, 'body > div.x1n2onr6.xzkaem6 > div.x9f619.x1n2onr6.x1ja2u2z > div > div.x1uvtmcs.x4k7w5x.x1h91t0o.x1beo9mf.xaigb6o.x12ejxvf.x3igimt.xarpa2k.xedcshv.x1lytzrv.x1t2pt76.x7ja8zs.x1n2onr6.x1qrby5j.x1jfb8zj > div > div > div > div > div.xb88tzc.xw2csxc.x1odjw0f.x5fp0pe.x1qjc9v5.xjbqb8w.x1lcm9me.x1yr5g0i.xrt01vj.x10y3i5r.xr1yuqi.xkrivgy.x4ii5y1.x1gryazu.x15h9jz8.x47corl.xh8yej3.xir0mxb.x1juhsu6 > div > article > div > div._aatk._aatl > div > div')
                            # screenshot_filename = f'Fooocus/outputs/refrence_images_from_insta/{account}_row{row_index}_post{post_index}.png'
                            large_photo.screenshot(path_to_save_images)
                            x_button = self.driver.find_element(By.CSS_SELECTOR, 'body > div.x1n2onr6.xzkaem6 > div.x9f619.x1n2onr6.x1ja2u2z > div > div.x160vmok.x10l6tqk.x1eu8d0j.x1vjfegm > div > div')
                            x_button.click()





                    last_row_index = len(rows_of_posts)
                    self.driver.execute_script("window.scrollBy(0, 1000);")  # Scroll down to load more posts
                    time.sleep(3)  # Wait for new posts to load

                self._search()
            except Exception as e:
                print('error', e)
                self._search()
                continue


    def preform(self, accounts_to_copy, path_to_save_images):
        time.sleep(10)
        self.login()
        time.sleep(10)
        self.copy_images(accounts_to_copy, path_to_save_images)









# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from openai import OpenAi
# driver = webdriver.Firefox()
# driver.get("https://dev.to")

# # Wait for the search input to be visible and then send keys
# search_input = WebDriverWait(driver, 10).until(
#     EC.visibility_of_element_located((By.CLASS_NAME, "crayons-header--search-input"))
# )
# search_input.send_keys("Selenium")

import webbrowser
webbrowser.open('https://inventwithpython.com/')

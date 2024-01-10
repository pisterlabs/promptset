from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from openai import OpenAI
import time
import sys

query_example='s23'
url = 'https://www.cnbc.com/search/?query='+query_example+'&qsearchterm='+query_example
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Function to wait for the page to load

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument('--headless')  # Run Chrome in headless mode (no GUI)

# Create a new instance of the Chrome driver
driver = webdriver.Chrome(options=chrome_options)

# Navigate to the URL using Selenium
driver.get(url)

# Wait for the page to load (you might need to adjust the sleep time or use more sophisticated waiting techniques)
time.sleep(17)

# Get the page source after it has loaded
page_source = driver.page_source

# Close the Selenium-driven browser
driver.quit()

# Now you can use BeautifulSoup to parse the loaded HTML content
soup = BeautifulSoup(page_source, 'html.parser')

# Extract data from the loaded page using BeautifulSoup
# For example, let's find all the links on the page
links = soup.find_all('a', attrs={'class': 'resultlink'})

new_links=[]
for link in links:
    if link.text:
        new_links.append(link.get('href'))
        print(link.get('href'))
        driver = webdriver.Chrome()
        driver.get(str(link.get('href')))
        time.sleep(5)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        article_title=soup.find('h1',attrs={'class':'ArticleHeader-headline'})
        if article_title.contains(query_example):
            client = OpenAI(api_key="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")
            stream = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "user", "content": article_title.text + "\nReview this headline and give a rating ranging from -100 to 100 on the impact this headline will have on the company, negative meaning bad impact and positive meaning good impact. Explain the rating in just a few words. IMPORTANT: first write the grade with the format 'Grade: your_grade'"}],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content,end="")
            print()
        driver.quit()

        # print(link.get('href'))
        # print(link.text)
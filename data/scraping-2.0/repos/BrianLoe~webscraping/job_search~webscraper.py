#!/usr/bin/env python
# coding: utf-8

from login import config
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import sys
import openai
import pandas as pd

# Import credentials
params = config()
openai.api_key = params['api_key']
system_msg="You are a career coach who understands job requirements"

def input_rec_url():
    ### A LinkedIn job search result page for user to provide.
    ### If not provided, it will use a provided job page link which is recommended page.
    
    recomend_url = input("Please enter linkedin job search result page link: ")
    # check url
    if not recomend_url.startswith('https://www.linkedin.com'):
        print("Job site is invalid")
        ans = input("would you like to use a provided link? [y/n]")
        flag = 'link'
        if ans.lower()=='y':
            recomend_url = "https://www.linkedin.com/jobs/collections/recommended?lipi=urn%3Ali%3Apage%3Ad_flagship3_job_home%3BobHSV5i9RA6yU%2F%2F5gLL1pQ%3D%3D"
        else:
            error_handler(ans, flag)
    return recomend_url

def input_num_pages():
    ### The number of pages the user would like to scrape.
    
    num_pages = input("Please enter the number of pages you would like to search (1-9): ")
    flag = 'num'
    if (not num_pages.isdigit()):
        ans = input("invalid number, would you like to try again? [y/n] ")
        error_handler(ans, flag)
    else:
        print("Starting program ...")
        return num_pages
    
def error_handler(input_arg, flag):
    if input_arg=='n':
        print("Exiting ...")
        sys.exit(1)
    elif flag=='num':
        input_num_pages()
    elif flag=='link':
        input_rec_url()
        
def create_webdriver():
    print("Creating webdriver instance ...")
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    return webdriver.Chrome(options=options)

def login(driver):
    print("Opening linkedIn's login page ...")
    driver.get("https://linkedin.com/uas/login")

    print("waiting for the page to load ...")
    time.sleep(5)

    print("entering username ...")
    username = driver.find_element(By.ID, "username")
    # In case of an error, try changing the element tag used here.

    # Enter Your Email Address
    username.send_keys(params['user'])

    print("entering password ...")
    pword = driver.find_element(By.ID, "password")
    # In case of an error, try changing the element Wtag used here.

    # Enter Your Password
    pword.send_keys(params['password'])

    print("Clicking on the log in button ...")
    # Format (syntax) of writing XPath --> //tagname[@attribute='value']
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    return

def start_scraping(driver, num_pages):
    print("Getting the job lists ...")
    # initialise dictionary to store all the information
    job_dict = {}
    job_dict['job_title'] = []
    job_dict['company'] = []
    job_dict['location'] = []
    job_dict['level'] = []
    job_dict['time_posted'] = []
    job_dict['graduate_suitable'] = []
    
    get_job_titles_company(driver, num_pages, job_dict)
    df = pd.DataFrame(job_dict)
    return df

def write_textfile(df):
    print("Creating text file ...")
    try:
        df.to_csv('result.txt', index=False)
        print("Text file created")
    except:
        print("Process was unsuccessful")
    return

def run_program(url, num_pages):
    # create chrome webdriver
    driver = create_webdriver()
    # This instance will be used to log into LinkedIn
    login(driver)
    # In case of an error, try changing the XPath used here.
    print("waiting for the page to load (please verify the human verification if exist) ...")
    time.sleep(15)
    
    print("Getting the recommended jobs page ...")
    driver.get(url)
    time.sleep(5)
    
    df = start_scraping(driver, num_pages)
    
    # Convert to text file
    write_textfile(df)

def gpt_message(msg):
    ### A function to ask GPT to provide answer as a career coach.
    ### It will return a response that answers the user message 'msg'
    response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "system", "content": system_msg},
                                            {"role": "user", "content": msg}]
                                )
    response = response["choices"][0]["message"]["content"]
    return response

def search_job_in_page(driver,job_list, cur_page_num, job_dict):
    ### A function to search for relevant information about a job.
    ### Each information is located in different element. This function scrapes each of the information using element and class.
    ### Returns the current page number which is being scraped.
    
    ### The job list which appears on the left-hand side of LinkedIn page.
    job_name = job_list.find_all('li', {'class':'ember-view'})
    for row in job_name:
        try:
            # Extract the left-hand side info
            print('Extracting job_title ...')
            job_title = row.find('a').contents[0].strip()
            print('Extracting company ...')
            company = row.find('div', {'class':'job-card-container__company-name'}).contents[0].strip()
            # id for the job specified.
            rowid = row.get('id')
            print('Clicking job card ...')
            # change focus on the right-hand side of page which provide the job card of job clicked.
            # the id is used to identify which job needs to be clicked.
            driver.find_element(By.XPATH, "//li[@id='"+rowid+"']").click()
            # Wait for the page to load
            time.sleep(3)
            # Extract the right-hand side info
            src = driver.page_source
            soup = BeautifulSoup(src, 'html5lib')
            job_card = soup.find('div', {'class':'scaffold-layout__list-detail-inner'})
            print('Extracting level ...')
            level = job_card.find('li', {'class':'jobs-unified-top-card__job-insight'}).find('span').contents[2]
            print('Extracting location ...')
            location = job_card.find('span', {'class':'jobs-unified-top-card__bullet'}).contents[0].strip()
            print('Extracting time_posted ...')
            time_posted = job_card.find('span', {'class':'jobs-unified-top-card__posted-date'}).contents[0].strip()
            print('Extracting job description ...')
            text = job_card.find('div', {'class':'jobs-description__content jobs-description-content'}).find('span').getText().strip()
            print('Converting job description ...')
            suit_msg = "Would the job suit a recent graduate based on this job description (answer with yes or no followed with brief explanation): "
            suit_msg+=text
            response = gpt_message(suit_msg)
            # store the info in dictionary
            job_dict['job_title'].append(job_title)
            job_dict['company'].append(company)
            job_dict['location'].append(location)
            job_dict['level'].append(level)
            job_dict['time_posted'].append(time_posted)
            job_dict['graduate_suitable'].append(response)
        # if job is not found anymore, move on to next page.
        except: 
            try:
                # this stores the page number that were found.
                p_num = int(row.find('button').attrs.get('aria-label')[-1:])
                # if current page number is lower, then we need to move on to next page.
                if cur_page_num<p_num:
                    cur_page_num=p_num
                    break
                # if current page number is higher, then we need to find next page number until the above condition is true.
                elif cur_page_num>=p_num:
                    continue
            except:
                continue
    return cur_page_num

def get_job_titles_company(driver, last_page, job_dict):
    ### Iterate until the last_page which is the number of pages user provided.
    ### This function will call the function to scrape all jobs within a page.
    ### If it reaches the end of page, it will continue to the next page until it reaches the number of pages user requested.
    for i in range(1, last_page+1):
        src = driver.page_source

        # Now using beautiful soup
        soup = BeautifulSoup(src, 'html5lib')

        # Find the element that stores the list of jobs including the job card.
        job_list = soup.find('div', {'class':'scaffold-layout__list-detail-inner'})

        next_page_num = search_job_in_page(driver,job_list,i, job_dict)

        # Click the next page button.
        driver.find_element(By.XPATH, "//button[@aria-label='Page "+str(next_page_num)+"']").click()
        # Wait for page to load
        time.sleep(6)
    return next_page_num

if __name__ == "__main__":
    recomend_url = input_rec_url()
            
    num_pages = int(input_num_pages())    
    run_program(recomend_url, num_pages)
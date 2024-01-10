

import requests
import json
import time
import openai
import re
import os
from dotenv import load_dotenv
import logging 

def search_jobs(query, site, num_results=10, day=1):
    job_listings = []
    load_dotenv()
    # API Key and Search Engine ID
    api_key = os.getenv('GOOGLE_API_ID')
    cx = os.getenv('GOOGLE_CX_ID')
    if api_key is None or cx is None:
        raise Exception('Failed to get Google API key or CX ID from environment variable')
    print("api_key: ", api_key)
    print(cx)
    date_restrict= "d" + str(day)
    
    # Loop through pages in multiples of 10 for pagination
    for start in range(0, num_results, 10):
        url = f'https://www.googleapis.com/customsearch/v1'
        
        params = {
            'q': query,    # query
            'key': api_key,    # api key
            'cx': cx,    # cx id
            'start': start+1,   # start index
            'dateRestrict': date_restrict,
            'siteSearch': site
        }

        # Request to the API
        res = requests.get(url, params=params)
        
        print(res.status_code)
        print(res.text)

        if res.status_code == 200:

            result_json = res.json()
            items = result_json.get('items', [])
            
            # Extract the link, title, and snippet from each item
            for item in items:
                link = item.get('link', '')
                title = item.get('title', '')
                snippet = item.get('snippet', '')
                job_listings.append({
                    'link': link,
                    'title': title,
                    'snippet': snippet
                })
        else:
            logging.error("something is wrong with google scraping on career site")
            break

    return job_listings




def filter_links(job_listings):
    grad_keywords = ['Engineeing', 'Software', 'Machine Learning', 'Engineer', 'Computer Vision', 'QA', 'Research', "AI", ]

    intern_keywords = ["Intern", "Internship", "Co-op", "Student", "Summer", 
                       "Semester", "Cooperative Education", "Work Study", 
                       "Part-Time", "Temporary", "Seasonal", "Trainee", 
                       "Undergraduate", "Research Assistant", "Fellow", 
                       "Pre-grad", "Pre-graduate", "Placement", 
                       "Work Placement", "Student Worker", "Apprentice", 
                       "Practicum"]

    grad_links = []
    intern_links = []

    for job in job_listings:
        title = job['title']
        link = job['link']
        snippet = job['snippet']
        if any(re.search(f"{keyword}", title, re.IGNORECASE) for keyword in grad_keywords):
            grad_links.append({"title": title, "link": link, "snippet":snippet})
        if any(re.search(f"{keyword}", title, re.IGNORECASE) for keyword in intern_keywords):
            intern_links.append({"title": title, "link": link, "snippet":snippet})

    return grad_links, intern_links








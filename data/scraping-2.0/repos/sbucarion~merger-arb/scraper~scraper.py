import re
import time
import requests
from bs4 import BeautifulSoup
import openai
import pdfkit
import sqlite3
import os 
from datetime import datetime

def get_all_filings(soup):
    #Go through all filings with html text and accerssion
    filings = []
    for row in soup.findAll("tr"):
        if "Accession Number" in row.text:
            filings.append(row)
            
    return filings


def get_acc_no(text):
    # Extract the accession number using regex
    match = re.search(r"Accession Number: (\d{10}-\d{2}-\d{6})", text)
    if match:
        accession_number = match.group(1)
        return (accession_number)
    
    
def get_filing_metadata(filing):
    for links in filing.findAll('a'):
        href = links['href']

        if ".htm" in href:
            #"click" on link (just request that link)
            x = requests.get(r"http://sec.gov" + href, headers=headers)

            
    soup = BeautifulSoup(x.text, "html.parser")
    cik = re.search(r"CIK:\s+(\d+)", soup.text).group(1)
    
    for _ in soup.findAll('a'):
        if "ix?doc=" in _['href']:
            partial_link = _['href'].split("/ix?doc=")[-1]

            filing_link = "http://sec.gov" + partial_link
            
            return filing_link, cik
        
        
def get_filing_time(filing):
    time_data = filing.findAll('td')[3]
    date = time_data.contents[0]
    time = time_data.contents[2]
    
    datetime_obj = datetime.strptime(date + " " + time, '%Y-%m-%d %H:%M:%S')
    unix_time = int(datetime_obj.timestamp())
    
    return unix_time
        
    
def get_filing(filing_link):
    raw_filing = BeautifulSoup(requests.get(filing_link, headers=headers).text, "html.parser").find("body").text
    filing = clean_filing(raw_filing)
    
    return filing
        
        
def clean_filing(raw_filing):
    filing = raw_filing.replace("\n", " ").replace("\xa0", " ").strip()
    filing = " ".join(filing.split())


    filing = "UNITED STATES SECURITIES AND EXCHANGE COMMISSION" + \
            filing.split("UNITED STATES SECURITIES AND EXCHANGE COMMISSION")[-1]
    
    return filing.lower()


def is_filing_merger(filing_text):
    #Need to make more solid determination
    if "merger" not in filing_text:
        return False
    
    if "item 1.01".lower() in filing_text and "item 7.01".lower() in filing_text:
        return True
    
    
    return False

    
headers = {
"User-agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36"
}

base_url = r"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK=&type=8-K&owner=include&count=100&action=getcurrent"


current_dir = os.getcwd() + "\\" 


path_wkhtmltopdf = current_dir.split("scraper")[0] + "wkhtmltopdf\\bin\\wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

DB_PATH = current_dir.split("scraper")[0] + "database\\filing_data.sqlite3"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()


while True:
    latest_8k_filings = requests.get(base_url, headers=headers).text
    soup = BeautifulSoup(latest_8k_filings, "html.parser")

    #Iterate through all filings on page
    filings = get_all_filings(soup)

    x = time.time()


    cursor.execute("SELECT accession_no, unix_number FROM seen_filings")
    seen = cursor.fetchall()
    all_accession_numbers = {row[0] for row in seen}
    max_unix_number = max({row[1] for row in seen})

    for filing in filings:
        filing_acc_no = get_acc_no(filing.findAll('td')[2].text)

        if filing_acc_no in all_accession_numbers:
            time.sleep(2)
            continue

        filing_link, company_cik = get_filing_metadata(filing)
        filing_time = get_filing_time(filing)


        if max_unix_number > filing_time:
            time.sleep(2)
            continue


        filing_text = get_filing(filing_link)

        if is_filing_merger(filing_text.lower()):
            print(True, filing_link)


            try:
                #Store metadata to db
                cursor.execute("INSERT INTO data (accession_no,cik, unix_number) VALUES (?, ?, ?)",
                               (filing_acc_no, company_cik, filing_time))

                # Commit the changes to the database
                conn.commit()

            except:
                time.sleep(2)
                continue


            #Save as pdf to 8k folder
            filings_path = current_dir.split("scraper")[0] + "8ks\\"
            filing_path = filings_path + filing_acc_no + ".pdf"
            pdfkit.from_url(filing_link, filing_path, configuration=config)


        try:
            cursor.execute("INSERT INTO seen_filings (accession_no, unix_number) VALUES (?, ?)",
                       (filing_acc_no, filing_time))

            conn.commit()

        except:
            time.sleep(2)
            continue


        print("Scanned: ", filing_link)
        print("_______________________________")
        print("\n")


    cursor.execute("SELECT accession_no, unix_number FROM seen_filings")
    seen = cursor.fetchall()
    all_accession_numbers = {row[0] for row in seen}
    max_unix_number = max({row[1] for row in seen})

    time.sleep(2)
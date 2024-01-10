import json
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
import requests
import re

from utils import *


INDUSTRY_KEYWORD = os.environ.get('INDUSTRY_KEYWORD')
WHOISJSONAPI= os.environ.get('WHOISJSONAPI')
data_folder = f"data/{INDUSTRY_KEYWORD}"
companies_file = "1companies.json"

targetField="domainRegDate"

def load_data_without_regdate(filename):
    return {domain: info for domain, info in load_from_json_file(filename,data_folder).items() if "domainRegDate" not in info and "nature" in info and info["nature"] == "single project"}


def main():
    # Define paths and filenames
    
    
    
    # Load data
    data = load_data_without_regdate(companies_file)

    # Iterate through the data dictionary
    for domain, domain_data in data.items():
        
        print(domain)

        # Define the endpoint and headers
        url = 'https://whoisjsonapi.com/v1/'+domain
        headers = {
            'Authorization': 'Bearer '+WHOISJSONAPI  # Replace {your token} with your actual token
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            req = response.json()
            req = req['domain']
            created_date = req.get('created_date', 'Date not found')
            print(f"Created Date: {created_date}")
        else:
            created_date = (f"Error {response.status_code}: {response.text}")

        data[domain][targetField] = created_date

        save_to_json_file(data, companies_file,data_folder)
        

if __name__ == "__main__":
    main()
    print("next:  5")
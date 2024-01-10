from dotenv import load_dotenv
import openai
import os
import json
import requests
from bs4 import BeautifulSoup

with open('backend/data/user_purchase_data.json', 'r') as json_file:
    user_data = json.load(json_file)
load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key

product_urls = []

for user in user_data:
    for purchase_history in user.get('purchase_history', []):
        if isinstance(purchase_history, str):
            product_urls.append(purchase_history)

def get_product_info(product_links):
    product_info_list = []
    
    for link in product_links:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")
        product_name = soup.find("span", {"class": "B_NuCI"})
        if product_name:
            product_name = product_name.get_text()
        product_other_info = soup.find("div",{"class":"_1AN87F"})
        if product_other_info:
            product_other_info = product_other_info.get_text()

        specifications={}
        details_div = soup.find("div", {"class": "X3BRps"})
        rows = details_div.find_all("div", {"class": "row"})
        for row in rows:
            cols = row.find_all("div", {"class": "col"})
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                specifications[key] = value

        product_info = {}
        product_info['name'] = product_name
        product_info['specs'] = specifications
        if product_other_info:
            product_info['others'] = product_other_info
        
        product_info_list.append(product_info)
    
    return product_info_list

def main():
    all_product_info = []    
    for url in product_urls:
        product_info = get_product_info([url])  
        all_product_info.extend(product_info)    

    formatted_input = "\n".join(json.dumps(product) for product in all_product_info)

    openai_data = {
        "model": "text-curie-001",
        "prompt": f"List 3-4 keywords that describe this user's style,preferences and likes from this para :\n\n{formatted_input}",
        "max_tokens": 50        
    }

    response = openai.Completion.create(**openai_data)
    generated_text = response.choices[0].text.strip()
    return generated_text

if __name__ == "__main__":
    main()

import requests
from bs4 import BeautifulSoup
from scrape_cashback_table import scrape_individual, scrape_img
import openai
import pymongo
import os
from dotenv import load_dotenv

from dotenv import load_dotenv, find_dotenv
import os
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

client = pymongo.MongoClient(f"mongodb+srv://{DB_USER}:{DB_PASS}@hentaicluster.hcelnlh.mongodb.net/?retryWrites=true&w=majority")
db = client["cardCalc"]
col = db["cashback"]
colImg = db["cashbackImg"]

url = "https://ringgitplus.com/en/credit-card/cashback/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

ul_tag = soup.find("ul", class_="Products CRCD")
li_tags = ul_tag.find_all("li")

links_and_names = {}

for li in li_tags:
    a_tag = li.find("a")
    link = "https://ringgitplus.com" + a_tag["href"]
    name = a_tag.text
    links_and_names[name] = link
    individual_table_rows = scrape_individual(link)
    img_src = scrape_img(link)
    
    seen_pair, seen_categories = set(), set()
    
    # if name is not in dbs, add it
    if col.find_one({"name": name}) == None:
        col.insert_one({"name": name, "link": link, "categories": []})
    if colImg.find_one({"name": name}) == None:
        colImg.insert_one({"name": name, "img_src": img_src})

    categories = []

    for row in individual_table_rows:
        # seen key: (categories, spend)
        if (tuple(row[0]), tuple(row[3])) in seen_pair:
            continue
        seen_pair.add((tuple(row[0]), tuple(row[3])))

        if tuple(row[0]) in seen_categories:
            for category in categories:
                if category["individual_categories"] == row[0]:
                    category["tier"].insert(0, {"cashback_percentage": row[1], "monthly_cap": row[2], "spend": row[3]})
        else:
            categories.append({"individual_categories": row[0], "tier": [
                {"cashback_percentage": row[1], "monthly_cap": row[2], "spend": row[3]}]})
        seen_categories.add(tuple(row[0]))
    
    col.update_one({"name": name}, {"$set": {"categories": categories}})
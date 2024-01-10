# Set env var OPENAI_API_KEY or load from a .env file:
# import dotenv
# dotenv.load_dotenv()

from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
import re
import json
import requests
import os

# Load HTML
loader = AsyncChromiumLoader(["https://rostender.info/extsearch/advanced?query=e9ac7cfa7191307db492aaf70b1b5cf6"])
html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html,tags_to_extract=["a","span"])

result = docs_transformed[0].page_content[0:2000000]

# Patterns to capture the tender details: Link and Region
pattern = re.compile(r'\((/tender/\d+)\)\s*([\w\s\-]+)')

# Using a set to track processed tenderIds
seen_tender_ids = set()
tenders = []
existing_tenders = []

if os.path.exists("tenders.json"):
    with open("tenders.json", "r", encoding='utf-8') as f:
        existing_tenders = json.load(f)
        for tender in existing_tenders:
            seen_tender_ids.add(tender["tenderId"])

matches = pattern.findall(result)

for match in matches:
    tender_id = match[0].split("/")[-1]
    if tender_id not in seen_tender_ids:
        seen_tender_ids.add(tender_id)

        tender_data = {
            "tenderId": tender_id,
            "region": match[1].strip(),
            "link": "https://rostender.info" + match[0]
        }
        tenders.append(tender_data)

        # Send data to webhook
        webhook_url = f"https://noxon.wpmix.net/counter.php?totenders=1&msg={json.dumps(tender_data, ensure_ascii=False)}"
        requests.get(webhook_url)

# Append new tenders to the list of existing tenders
existing_tenders.extend(tenders)

# Save to JSON
with open("tenders.json", "w", encoding='utf-8') as f:
    json.dump(existing_tenders, f, ensure_ascii=False, indent=4)

# Display the count of new tenders found
print(f"Total new tenders found: {len(tenders)}")
for tender in tenders:
    print(f"Tender ID: {tender['tenderId']}, Region: {tender['region']}, Link: {tender['link']}")

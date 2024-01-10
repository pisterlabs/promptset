from langchain.document_loaders import PyPDFLoader

import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

url = "https://www.govinfo.gov/app/collection/chrg/118/house/Committee%20on%20Agriculture"

#If there is no such folder, the script will create one automatically
folder_location = r'./webscraping'
if not os.path.exists(folder_location):os.mkdir(folder_location)

response = requests.get(url)
soup=BeautifulSoup(response.text, parser='html.text')
# print(list(soup.descendants))
print(soup.descendants.select("a[href$='118']"))
# for link in soup.select("a[href$='.pdf']"):
#     #Name the pdf files using the last portion of each link which are unique in this case
#     filename = os.path.join(folder_location,link['href'].split('/')[-1])
#     with open(filename, 'wb') as f:
#         f.write(requests.get(urljoin(url,link['href'])).content)


# loader = PyPDFLoader("https://www.govinfo.gov/content/pkg/CHRG-118hhrg52557/pdf/CHRG-118hhrg52557.pdf")
# data = loader.load()
# print(data)
from langchain.document_loaders import WebBaseLoader
import requests
import xml.etree.ElementTree as ET


sitemap_url = "https://www.fwd.com.vn/sitemap-pages.xml"
response = requests.get(sitemap_url)

if response.status_code == 200:
    tree = ET.fromstring(response.content)
    loc_elements = tree.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")

    list_url = [loc_element.text for loc_element in loc_elements]
    print(len(list_url))

    loader = WebBaseLoader(list_url)
    docs = loader.load()

else:
    print("Failed to retrieve sitemap. Status code:", response.status_code)

import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import UnstructuredURLLoader
from langchain_core.documents.base import Document
from urllib.parse import urlparse

# url = input("Insert Link That You Want to Scrape:")

def scrape_cnn(url):
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        result = soup.find_all(class_="detail-wrap flex gap-4 relative")

        # Clean up and concatenate the text using a for loop
        cleaned_text_list = []
        for element in result:
            cleaned_text = element.get_text().replace('\n', '').strip()
            cleaned_text_list.append(cleaned_text)

        # Join the cleaned text from the list
        all_text = " ".join(cleaned_text_list)

        # # Print or use the cleaned and concatenated text
        # print(all_text)

        # # Write the result to a text file
        # with open("result.txt", "w", encoding="utf-8") as f:
        #     f.write(all_text)

        return all_text
    else:
        print(f"Failed to retrieve the webpage. Status Code: {response.status_code}")

def scrape_kompas(url):
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            script_text = script.get_text()
            if "var keywordBrandSafety" in script_text:
                result = script_text
        result = result.replace ("var keywordBrandSafety =", "").strip().strip('"').strip('";')
        return result
    else:
        print(f"Failed to retrieve the webpage. Status Code: {response.status_code}")
  
def scrape_detik(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all(class_='detail__body-text itp_bodycontent')
        # Extract and return the text from each element
        cleaned_text_list = []
        for element in results:
            text = element.get_text().replace('\n', '').strip()
            cleaned_text_list.append(text)
        
        # Join the cleaned text from the list
        all_text = " ".join(cleaned_text_list)

        return all_text
    else:
        print(f"Failed to retrieve the webpage. Status Code: {response.status_code}")

def document_instance(link, content):
    document_instance = Document(
        metadata= {'source':link},
        page_content=content
    )
    return document_instance

def scrape_cnn_instance(url):
    content = scrape_cnn(url)
    return (document_instance(url, content))

def scrape_kompas_instance(url):
    content = scrape_kompas(url)
    return (document_instance(url, content))

def scrape_detik_instance(url):
    content = scrape_detik(url)
    return (document_instance(url, content))

def scraping_pipeline(links:list):
    result = []
    for link in links:
        parsed_url = urlparse(link)
        domain = parsed_url.netloc

        # filter for detik links
        if "detik.com" in domain:
            result.append(scrape_detik_instance(link))

        # filter for cnn
        elif "cnnindonesia.com" in domain:
            result.append(scrape_cnn_instance(link))
        
        # filter for kompas
        elif "kompas.com" in domain:
            result.append(scrape_kompas_instance(link))
        
        else:
            print(f"Failed to retrieve the webpage. because your domain was {domain}")
    return result

def langchain_url(url):
    loader = UnstructuredURLLoader([url])
    data = loader.load()
    return data


links = [
    'https://www.cnnindonesia.com/ekonomi/20231221152333-78-1040259/rupiah-merosot-ke-rp15525-jelang-rilis-data-inflasi-as',
    'https://www.cnnindonesia.com/olahraga/20231221131224-142-1040147/mohamed-salah-vs-arsenal-tajam-dan-lebih-sering-menang',
    'https://finance.detik.com/infrastruktur/d-7101502/ini-bocoran-konglomerat-yang-bakal-susul-aguan-cs-investasi-di-ikn'
]

if __name__ == "__main__":
    print(scraping_pipeline(links =links))

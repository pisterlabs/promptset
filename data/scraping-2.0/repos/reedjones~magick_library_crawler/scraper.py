__author__ = "reed@reedjones.me"
"""
Todo - convert PDFs to images where text extraction doesn't work then use OCR to convert images to text 
"""

import logging
import os
import pickle
import time
from collections import Counter
from io import StringIO, BytesIO
from urllib.request import urlopen, Request
from zipfile import ZipFile
import pandas as pd
import bs4
import langid
import pdf2image
import pytesseract
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import PyMuPDFLoader as MyPDFLoader
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from requests_ip_rotator import ApiGateway
from datastore import load, dump, finished_url, problem_url, store_data, append_to_aws, load_from_aws
import uuid
already_on_s3 = "processed_items.pickle"
already_scraped = "already_scraped.pickle"
uid = uuid.uuid4()
import boto3
import json
from datetime import date
from ocr import url_to_text
finished = f'finished_{uid}.pickle'
problem = f'problem_{uid}.pickle'
results = f'results_{uid}.pickle'

import logging
logging.basicConfig(filename='scraper.log', encoding='utf-8', level=logging.DEBUG)



# http_client.HTTPConnection.debuglevel = 1
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.WARNING)
requests_log.propagate = True

resource_manager = PDFResourceManager()


def check_pdf_is_parseable(url, session):
    fake_file_handle = StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)

    response = get_with_session(url, session=session)
    fb = BytesIO(response.content)

    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    parser = PDFParser(fb)
    doc = PDFDocument(parser, password='', caching=True)
    # Check if the document allows text extraction.
    # If not, warn the user and proceed.
    extractable = doc.is_extractable
    fb.close()
    converter.close()
    fake_file_handle.close()
    return extractable


def extract_text_from_pdf_url(url):
    fake_file_handle = StringIO()
    converter = TextConverter(resource_manager, fake_file_handle)

    request = Request(url)
    response = urlopen(request).read()
    logging.debug(f"Got response {response}")
    fb = BytesIO(response)

    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    parser = PDFParser(fb)
    doc = PDFDocument(parser, password='', caching=True)
    # Check if the document allows text extraction.
    # If not, warn the user and proceed.
    if not doc.is_extractable:
        return None
    for page in PDFPage.get_pages(fb, caching=True, check_extractable=True):
        page_interpreter.process_page(page)

    text = fake_file_handle.getvalue()

    # close open handles
    fb.close()
    converter.close()
    fake_file_handle.close()

    if text:
        # If document has instances of \xa0 replace them with spaces.
        # NOTE: \xa0 is non-breaking space in Latin1 (ISO 8859-1) & chr(160)
        text = text.replace(u'\xa0', u' ')

        return text


def text_from_url(u, second=1):
    logging.debug("waiting")
    if not u:
        return ""
    loader = MyPDFLoader(u)
    try:
        data = loader.load()
        logging.debug(f"Got text {data[0]}")
        return data[0]
    except Exception as e:
        logging.debug(e)
        if second > 3:
            return ""
        time.sleep(5 + second * 3)
        return text_from_url(u, second=second + 1)



def text_from_url2(u):
    loader = MyPDFLoader(u)
    try:
        data = loader.load()
        logging.debug(f"Got text {data[0]}")
        return data[0]
    except Exception as e:
        logging.debug(e)


def images_to_txt(path, language):
    images = pdf2image.convert_from_bytes(path)
    all_text = []
    for i in images:
        pil_im = i
        text = pytesseract.image_to_string(pil_im, lang=language)
        # ocr_dict = pytesseract.image_to_data(pil_im, lang='eng', output_type=Output.DICT)
        # ocr_dict now holds all the OCR info including text and location on the image
        # text = " ".join(ocr_dict['text'])
        # text = re.sub('[ ]{2,}', '\n', text)
        all_text.append(text)
    return all_text, len(all_text)


def convert_pdf_to_txt_pages(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    size = 0
    c = 0
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()
        if c == 0:
            texts.append(t)
        else:
            texts.append(t[size:])
        c = c + 1
        size = len(t)
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return texts, nbPages


def convert_pdf_to_txt_file(path):
    texts = []
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
        interpreter.process_page(page)
        t = retstr.getvalue()
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return t, nbPages


def save_pages(pages):
    files = []
    for page in range(len(pages)):
        filename = "page_" + str(page) + ".txt"
        with open("./file_pages/" + filename, 'w', encoding="utf-8") as file:
            file.write(pages[page])
            files.append(file.name)

    # create zipfile object
    zipPath = './file_pages/pdf_to_txt.zip'
    zipObj = ZipFile(zipPath, 'w')
    for f in files:
        zipObj.write(f)
    zipObj.close()

    return zipPath


def extract_books_data(url):
    # Send a GET request to the website
    response = requests.get(url)

    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the book elements on the page
    book_elements = soup.find_all('div', class_='book')

    # Initialize a variable to store the total data size
    total_data_size = 0

    # Iterate over each book element and extract the title and file size
    for book_element in book_elements:
        # Find the title element within the book element
        title_element = book_element.find('div', class_='title')
        title = title_element.text.strip()  # Extract the text and remove leading/trailing whitespace

        # Find the file size element within the book element
        file_size_element = book_element.find('div', class_='filesize')
        file_size = file_size_element.text.strip()  # Extract the text and remove leading/trailing whitespace

        # Add the file size to the total data size
        total_data_size += int(file_size)

        # logging.debug the title and file size
        logging.debug('Title:', title)
        logging.debug('File Size:', file_size)
        logging.debug('---')

    # logging.debug the total data size
    logging.debug('Total Data Size (bytes):', total_data_size)


# URL of the website
start_url = 'http://english.grimoar.cz/?Loc=key&Lng=2&S='  # 0
end_url = 'http://english.grimoar.cz/?Loc=key&Lng=2&S='  # 2011


def get_page(num):
    return f"http://english.grimoar.cz/?Loc=key&Lng=2&S={num}"


def all_pages():
    for i in range(198, 2011): # last was 198
        yield get_page(i)


# Call the function to extract the books data
# extract_books_data(start_url)


def get_page_html(url):
    """
    Function to get the HTML content of a webpage given its URL.
    """
    response = requests.get(url)
    html_content = response.text
    return html_content


def is_download_link(tag):
    return tag.name == 'a' and tag.has_attr('href') and 'download' in tag.text


def get_book_links(url):
    html_content = get_page_html(url)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Scrape the books data from the current page
    links = []
    next_page_link = soup.find('a', class_='next')
    if next_page_link:
        next_page_url = next_page_link.get('href')
        # Recursive call to get books data from the next page
        links += get_book_links(next_page_url)


# nnumber, link() with title, authors, maybe size, date
def clean_file_size(v):
    return v.replace(" ", "").strip().replace("(", "").replace(")", "").replace('.', '').strip()[:-1]


def has_keyword_str(t):
    return t.name == 'h4' and t.text == "Keywords suggestions"


def parse_keywords(soup):
    tag = soup.find('h4', string="Keywords suggestions")
    keywords = []
    if not tag:
        return keywords
    while True:
        if isinstance(tag, bs4.element.Tag):
            if tag.name == 'form':
                break
            elif tag.name == 'a':
                keywords.append(tag.text)
        tag = tag.nextSibling
    return keywords


def get_keywords(u):
    if not u:
        return []
    try:
        html_content = get_page_html(u)
        soup = BeautifulSoup(html_content, 'html.parser')
        return parse_keywords(soup)
    except Exception as e:
        logging.debug(e)
        return []


def test_keywords():
    assert os.path.isfile('keyword.html')
    with open('keyword.html') as d:
        soup = BeautifulSoup(d.read(), 'html.parser')
    logging.debug(soup)
    keywords = parse_keywords(soup)
    logging.debug(keywords)
    return keywords


def try_with_default(default, fun, fun_param):
    try:
        return fun(fun_param)
    except Exception as e:
        logging.debug(e)
        return default

def marked_scraped(data):
    store_data(data['title'], results=already_scraped, unique=True)

def mark_ons3(data):
    store_data(data['title'], results=already_on_s3, unique=True)

def check_on_s3(data):
    items = load(already_on_s3)
    return data['title'] in items
def check_scraped(data):
    items = load(already_scraped)
    return data['title'] in items




def scrape_table(table):
    column_names = ('number', 'title', 'author', 'size', 'date')
    rows = table.find_all('tr')
    table = []
    for row in rows:
        columns = row.find_all('td')
        link = columns[1].find('a', href=True)['href']
        columns = [i.text for i in columns]
        assert len(columns) == len(column_names)
        data = dict(zip(column_names, columns))

        if not check_scraped(data):
            marked_scraped(data)
            data['link'] = link
            data['size'] = clean_file_size(data['size'])
            try:
                data['lang'] = langid.classify(data['title'])[0]
            except Exception as e:
                data['lang'] = '?'
                logging.debug(e)

            data['download_url'] = book_url_to_download_url(data['link'])
            data['keywords'] = try_with_default([], get_keywords, data['link'])
            table.append(data)
        else:
            logging.debug(f"Skipping {data['title']}")

    return table


def get_container(soup):
    div = soup.find('div', id='right_inner')
    if div:
        return div.find('div', class_='margin')


def scrape_page(current_url):
    html_content = get_page_html(current_url)
    soup = BeautifulSoup(html_content, 'html.parser')
    container = get_container(soup)
    if container:
        table = container.find('table')
        if table:
            try:
                data = scrape_table(table)
                finished_url(current_url)
                return data
            except Exception as e:
                logging.debug(e)
    problem_url(current_url)


def book_url_to_download_url(url):
    if "Loc=book" in url:
        return url.replace("Loc=book", "Loc=dl")
    return None


def get_download_link(url):
    html_content = get_page_html(url)
    soup = BeautifulSoup(html_content, 'html.parser')
    content = soup.find('div', id='content')
    cont = content.find('div', id='cont')


def get_book_data(url):
    """
    Recursive function to scrape the books data from the website.
    It handles pagination and navigates through all the pages.
    """
    html_content = get_page_html(url)
    soup = BeautifulSoup(html_content, 'html.parser')

    # Scrape the books data from the current page
    books = []
    book_elements = soup.find_all('div', id='content')
    for book_element in book_elements:
        title = book_element.find('h2').text
        file_size = book_element.find('div', class_='fr').text
        books.append({'title': title, 'file_size': file_size})
        link = book_element.find('a', is_download_link)
        if link:
            link = link['href']

    # Check if there is a next page
    next_page_link = soup.find('a', class_='next')
    if next_page_link:
        next_page_url = next_page_link.get('href')
        # Recursive call to get books data from the next page
        # books += get_books_data(next_page_url)

    return books


def test_clean_file():
    t = "(32.657.206 B)"
    t2 = " (887.837 B)"
    logging.debug(clean_file_size(t))
    logging.debug(clean_file_size(t2))


def calculate_library_size(books):
    """
    Function to calculate the total amount of data the library contains in bytes.
    """
    total_size = 0
    for book in books:
        file_size = book['file_size']
        if 'KB' in file_size:
            size_in_bytes = int(file_size[:-2]) * 1024
        elif 'MB' in file_size:
            size_in_bytes = int(file_size[:-2]) * 1024 * 1024
        elif 'GB' in file_size:
            size_in_bytes = int(file_size[:-2]) * 1024 * 1024 * 1024
        else:
            size_in_bytes = int(file_size[:-1])
        total_size += size_in_bytes

    return total_size


def get_everything():
    for page in all_pages():
        logging.debug(f"Scraping {page}")
        data = scrape_page(page)
        if data:
            store_data(data, results=results)
            logging.debug("done")

        else:
            logging.debug("no data")




def step_2(data_name):
    data = load(data_name)
    data = flatten(data)
    for item in data:
        if isinstance(item['lang'], tuple):
            item['lang'] = item['lang'][0]





# Define the starting URL
starting_url = 'http://english.grimoar.cz/?Loc=key&Lng=2'


def get_langs_from_data():
    data = load(results)
    logging.debug(f"Data is {len(data)} items \n type: {type(data)} {type(data[0])} {len(data[0])}")
    # sizes = [int(i['size'])[:-1] for i in data]
    # logging.debug(f"{sum(sizes)}")
    logging.debug(data[0])
    total = 0
    langs = []
    for i in data:
        for item in i:
            item['lang'] = langid.classify(item['title'])
            if item['lang'] not in langs:
                langs.append(item['lang'])
    store_data(data)
    logging.debug(f"done\n {langs}")
    dump('langs.pickle', langs)
    lang_dict = dict.fromkeys(langs, 0)
    # for i in data:
    #     c = Counter(i.keys())
    #     logging.debug(c)
    #     for k, v in groupby(i, key=itemgetter('lang')):
    #         logging.debug(f"\t{k} -\n\t\t{v}\n")


# Scrape the books data

# Calculate the library size
# library_size = calculate_library_size(books_data)
def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def clean_lang():
    data = load(results)
    langs = load('langs.pickle')
    data = list(flatten(data))
    for i in data:
        i['lang'] = i.get('lang', ['?'])[0]
    c = Counter([d['lang'] for d in data])
    store_data(data)  # bad
    logging.debug(c)


def fetch_with_proxy(url, user_agent=None):
    if not user_agent:
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'

    headers = {'User-Agent': user_agent}
    with ApiGateway("http://english.grimoar.cz") as g:
        session = requests.Session()
        session.headers = headers
        session.mount("http://english.grimoar.cz", g)

        response = session.get(url)
        logging.debug(response.status_code)
        return response


def run_with_proxy(func, user_agent=None):
    if not user_agent:
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'

    headers = {'User-Agent': user_agent}
    with ApiGateway("http://english.grimoar.cz") as g:
        session = requests.Session()
        session.headers = headers
        session.mount("http://english.grimoar.cz", g)
        return func(session=session)


def run_with_proxy2(func, user_agent=None):
    if not user_agent:
        user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'

    headers = {'User-Agent': user_agent}
    with ApiGateway("http://english.grimoar.cz") as g:
        session = requests.Session()
        session.headers = headers
        session.mount("http://english.grimoar.cz", g)
        return func(session)


def get_with_session(url, **kwargs):
    session = kwargs.pop('session', None)
    if not session:
        raise Exception("Use session!")

    return session.get(url, **kwargs)


def check_files():
    with open("results_3de989d7-d32c-4465-827a-6e46c9ca52fa.pickle", 'rb') as f:
        data = pickle.load(f)
    # logging.debug(data)
    logging.debug(f"""
    Data : {len(data)}
    type: {type(data)}
    item 1 : {type(data[0])}
  
    """)
    titles = {


    }
    for item in flatten(data):
        t = item['title']
        count = titles.get(t, 0)
        count += 1
        titles[t] = count

    for k,v in titles.items():
        logging.debug(f"Title {k} \n #number items {v}")
    # data = [i for i in flatten(data)]
    # logging.debug(f"""
    #     Data : {len(data)}
    #     type: {type(data)}
    #     item 1 : {type(data[0])}
    #
    #     """)
    # if not os.path.isfile(results):
    #     logging.debug(f'will save to results')
    #     store_data(data)

def test_load():
    data = [{'title':'hello'}, {'title':'world'}]
    marked_scraped(data[0])
    marked_scraped(data[1])
    logging.debug("marked \n loading")
    data2 = load(already_scraped)
    logging.debug(data2)
    logging.debug("-------")
    logging.debug(check_scraped({'title':'hello'}))
    logging.debug(check_scraped({'title':'goat'}))
    marked_scraped({'title':'goat'})
    data3 = load(already_scraped)
    logging.debug(data3)
from timeit import default_timer as timer


def get_the_data(result_file):
    data = load(result_file)
    for item in flatten(data):
        yield item

def get_document_texts(result_file):
    for item in get_the_data(result_file):
        if not check_on_s3(item):
            target = item['download_url']
            doc = url_to_text(target)
            item['document_text'] = doc
            df = pd.DataFrame([item])
            append_to_aws(df)
            mark_ons3(item)
        else:
            logging.debug(f"Skipping {item['title']}")



def test_append_s3():
    data = [{'name':'reed'}]
    df = pd.DataFrame(data)
    append_to_aws(df)
    logging.debug("load")
    data2 = load_from_aws()
    logging.debug(data2)
    n = [{'name':'jo'}, {'name':'mama'}]
    nd = pd.DataFrame(n)
    append_to_aws(nd)
    data3 = load_from_aws()
    logging.debug(data3)




if __name__ == '__main__':
    logging.debug(f"Results will be stored in {results}")
    # get_everything()
    get_document_texts("results_a9fd713a-906a-43d5-9776-0f137fd2d3e2.pickle")

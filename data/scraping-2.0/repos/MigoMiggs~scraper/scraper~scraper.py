import time
from config import settings
from config.settings import Config
import langchain.embeddings
from urllib.parse import urljoin, urlparse
from utilities import utils
import requests
import os
from bs4 import BeautifulSoup
from vectorstore import vectorstore
from xml.etree import ElementTree as ET
from utilities.utils import (
    write_text_to_file,
    get_urls_from_sitemap,
    url_to_filename
)

class Scraper:
    """
    This class is responsible for scraping a website and storing the text and its embeddings in the vector database
    """

    # Class variables
    embeddings: langchain.embeddings = None
    config: settings.Config
    dict_done_urls: set
    fileParsed = None
    total_urls_scraped = 0
    _logger = None

    # Filenames
    FILE_ERRORS = './errors.txt'
    FILE_SCRAPED = './coo_urls_scraped.txt'

    DIR_SCRAPED = ''
    DIR_CHROMA = ''

    def __new__(cls, config_file_path='./config.yaml'):
        return super().__new__(cls)

    def __init__(self, config_file_path='./config.yaml'):
        """
        Constructor
        """

        # Laod the configuration to be available to the whole system
        self.config = Config(config_file_path)

        # Set config variables
        self.DIR_SCRAPED = self.config.data['site']['scraped-path']
        self.DIR_CHROMA = self.config.data['vectordb']['chroma-path']

        # if the directory self.DIR_SCRAPED does not exist, create it
        if not os.path.exists(self.DIR_SCRAPED):
            os.makedirs(self.DIR_SCRAPED)

        # if the directory self.DIR_CHROMA does not exist, create it
        if not os.path.exists(self.DIR_CHROMA):
            os.makedirs(self.DIR_CHROMA)

        # Set the logger
        if Scraper._logger is None:
            Scraper._logger = self.config.getLogger()

        self._logger.debug("Remove url files from previous runs")
        if os.path.exists(self.FILE_ERRORS):
            # If it exists, delete it
            os.remove(self.FILE_ERRORS)

        if os.path.exists(self.FILE_SCRAPED):
            os.remove(self.FILE_SCRAPED)

        # initialize other objects
        self.dict_done_urls = set()

    def get_config(self):
        """
        Returns the configuration
        :return: the configuration
        """
        return self.config

    def fetch_pdf(self, url: str, destination: str) -> None:
        """ Get a PDF from a URL and persist it as a file

        :param url:
        :param destination:
        :return:
        """
        self._logger.debug("Fetch PDF: " + url)
        try:
            response = requests.get(url)
            with open(destination, 'wb') as output_file:
                output_file.write(response.content)

        except Exception as ex:
            print('Could not write the following PDF: ' + url)
            print(ex)

    def extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extracts the text from the page, trying to ignore the header, footer, and other non-text elements
        :param soup:
        :return:
        """

        self._logger.debug("Extract text")
        # List of tags to be removed directly
        for tag in soup(['script', 'style', 'meta', 'link', 'noscript', 'cdata']):
            tag.decompose()

        # Class names that indicate elements to be removed
        classes_to_remove = ['footer', 'header', 'nav', 'aside', 'sidebar', 'menu']
        for class_name in classes_to_remove:
            for div in soup.find_all("div", class_=class_name):
                div.decompose()

        # Get the text from the page
        items = [item.text for item in soup.select('p, ol li')]
        # turn items into string
        text = ' '.join(items)

        # Removing extra spaces and joining text
        cleaned_text = '\n'.join(line.strip() for line in text.strip().splitlines() if line.strip())
        return cleaned_text

    def scrape_site(self, domain) -> None:
        """
        Scrapes the given domain
        :param domain:
        :return:
        """

        self._logger.debug("Scrape site")
        sitemap_list = self.fetch_sitemaps(domain)

        final_urls_to_scrape = {}

        # make sure domain is in the list
        final_urls_to_scrape[domain] = domain

        # Get the list of URL from website
        for sitemap in sitemap_list:
            urls_to_scrape = get_urls_from_sitemap(sitemap)
            final_urls_to_scrape.update(urls_to_scrape)

        # Get urls as a list
        keys = final_urls_to_scrape.keys()

        # log the total number of urls scraped
        self._logger.debug(f'========= total urls to scrape {len(keys)}')

        self.fileErrors = open(self.FILE_ERRORS, 'a')
        self.fileParsed = open(self.FILE_SCRAPED, 'a')

        for i in keys:
            self._logger.debug(f'========= total scraped {self.total_urls_scraped}')

            try:
                self.scrape(i)
            except Exception as e:
                self._logger.debug('ERROR Could not scrape' + str(i))
                self._logger.debug(e)
                self.fileErrors.writelines([i])

        self._logger.debug('done ' + str(self.total_urls_scraped))

        for url in self.dict_done_urls:
            self.fileParsed.write(url + "\n")

        self.fileErrors.close()
        self.fileParsed.close()

    def fetch_sitemaps(self, domain) -> list:
        """
        Fetches the sitemap.xml files from the given domain
        :param domain:
        :return:
        """

        robots_url = f"{domain}/robots.txt"
        sitemap_urls = []
        try:
            # Step 1: Fetch the robots.txt file
            response = requests.get(robots_url)
            response.raise_for_status()  # raise an error if the response contains an HTTP error status code
            lines = response.text.splitlines()

            # Step 2: Extract the list of sitemaps.xml files
            for line in lines:
                if line.startswith("Sitemap:"):
                    sitemap_url = line.split("Sitemap:")[1].strip()
                    sitemap_urls.append(sitemap_url)

            # If no sitemaps were found, add the default sitemap.xml
            if len(sitemap_urls) == 0:
                sitemap_urls.append(f"{domain}/sitemap.xml")

            # Step 3: Iterate through the list and open each file
            all_sitemaps = []
            for sitemap_url in sitemap_urls:
                response = requests.get(sitemap_url)
                if response.status_code != 200:
                    continue

                root = ET.fromstring(response.content)

                # Step 4: Check if it's a sitemapindex or a sitemap
                if root.tag.endswith("sitemapindex"):
                    # Step 5: If it's a sitemapindex, extract each sitemap
                    for sitemap in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
                        loc = sitemap.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                        if loc is not None:
                            all_sitemaps.append(loc.text)
                else:
                    all_sitemaps.append(sitemap_url)

            # Step 6: Return the list of sitemap.xml urls
            return all_sitemaps

        except requests.RequestException as e:
            print(f"Error fetching data from {domain}: {e}")
            return []


    def should_skip_file(self, file_extension: str) -> bool:
        """
        Checks if the file extension is in the list of non-text file extensions
        :param file_extension:
        :return:
        """

        self._logger.debug(f'File extension: {file_extension}')
        file_extension = file_extension.lower()

        # List of non-text file extensions to skip
        non_text_extensions = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff',  # Images
            '.zip', '.rar', '.tar', '.gz', '.7z',  # Archives
            '.mp3', '.wav', '.ogg', '.m4a', '.flac',  # Audio
            '.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv',  # Video
            '.xls', '.xlsx', '.ods',  # Spreadsheets
            '.ppt', '.pptx', '.odp'
            # ... add other non-text extensions here as needed
        }

        # Skip if the file extension is in the list of non-text extensions
        if file_extension in non_text_extensions:
            return True

        return False

    def scrape(self, url, visited=None, referer=None) -> None:
        """
        Scrapes the given URL, and it crawls the URL via the links within the page
        :param url: the URL to scrape
        :param visited: this allows us to recursively crawl
        :return:
        """

        if visited is None:
            visited = set()  # Set to keep track of visited URLs

        if url in self.dict_done_urls:
            self._logger.debug(f'URL already done, skipping: {url}')
            return

        self._logger.debug(f'About to scrape: {url} Number: {len(self.dict_done_urls)}')
        usevecrtordb = self.config.data['site']['use-vector-db']

        # Parse the original URL to get the domain
        original_domain = urlparse(url).netloc
        self._logger.debug(f'Scraping {url}')

        # Transform the URL to a filename
        transformed_url = url_to_filename(url)
        self._logger.debug(f'Transformed {transformed_url}')

        # Add the URL to the set of visited URLs
        visited.add(url)  # Mark the URL as visited
        self.dict_done_urls.add(url)

        # Get the filename and extension
        filename, filename_without_extension, file_extension = utils.get_filename_and_extension(url)

        # Skip if the file extension is not a text file
        if self.should_skip_file(file_extension):
            self._logger.debug(f'------ Skipping file: {filename}')
            return

        # If the file is .pdf, then fetch it and store it
        if file_extension == '.pdf':
            dir_filename = os.path.join(self.DIR_SCRAPED, filename)

            # check if dir_filename exists, fetch only if it does not
            if not os.path.isfile(dir_filename):
                self.fetch_pdf(url, dir_filename)
                time.sleep(1)

            # Store the text and its embeddings in the vector database
            if usevecrtordb:
                vectorstore.VectorDB().split_embed_store(dir_filename, '.pdf')
            return


        # Fetch the page
        try:
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36',
            }

            if referer is not None:
                headers['Referer'] = referer

            response = requests.get(url, headers=headers)
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Failed to retrieve {url}: {e}")
            return

        # If the fetch failed, log the error and return
        if response.status_code >= 400:
            self._logger.error(f"Failed to retrieve {url}: {response.status_code}")
            return

        # If fetch was successful, parse the page with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the text from the page
        url_text = self.extract_text(soup)
        if (url_text is None) or (url_text == ''):
            self._logger.debug(f'No text found for {url}')
            return

        # Write scraped text to file
        write_text_to_file(os.path.join(self.DIR_SCRAPED, transformed_url + '.txt'), url_text)

        # Store the text and its embeddings in the vector database
        if usevecrtordb:
            vectorstore.VectorDB().split_embed_store(os.path.join(self.DIR_SCRAPED, transformed_url + '.txt'), '.txt')

        # Extract all links
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']

            # Resolve relative links to absolute
            link = urljoin(url, link)

            # Normalize the new URL
            link = utils.normalize_url(link)

            # Parse the new URL to get its domain
            link_domain = urlparse(link).netloc

            # Skip already visited links, and limit to same domain
            if link not in visited and link_domain == original_domain:
                self.scrape(link, visited, referer=url)  # Recursively scrape each link



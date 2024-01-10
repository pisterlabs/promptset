import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from file_handler import rename_file, move_file, convert_to_epub
from libgen_api import LibgenSearch
import requests
from bs4 import BeautifulSoup
from icecream import ic
from zlibrary import Zlibrary
from isbntools.app import isbn_from_words
from isbnlib import meta



class BookScraper:
    """
    A class that provides functionality to scrape and download books from Libgen.
    """

    MIRROR_SOURCES = ["GET"]
    MIRROR_LIST = ['Mirror_1', 'Mirror_2', 'Mirror_3']

    def __init__(self):
        """
        Initializes the BookScraper object.
        """
        ic.configureOutput(includeContext=True)
        load_dotenv()  # This loads the .env file
        self._client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self._download_dir = "Downloads/"
        self._books_dir = "Books/"
        self._options = webdriver.ChromeOptions()
        self._driver = None
        self._Z = Zlibrary(email=os.getenv("GMAIL"),password=os.getenv("ZLIBRARY_PASSWORD"))   
    
    def _enable_download_headless(self):
        """
        Enable headless download in Chrome.
        """
        self._driver.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
        params = {'cmd':'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': self._download_dir}}
        self._driver.execute("send_command", params)

    def _initialize_driver(self):
        """
        Initialize the Chrome webdriver.
        """
        self._options.add_argument("--headless")
        self._options.add_argument("--window-size=1920x1080")
        self._options.add_argument("--disable-notifications")
        self._options.add_argument('--no-sandbox')
        self._options.add_argument('--verbose')
        self._options.add_experimental_option("prefs", {
            "download.default_directory": self._download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing_for_trusted_sources_enabled": False,
            "safebrowsing.enabled": False
        })
        self._options.add_argument('--disable-gpu')
        self._options.add_argument('--disable-software-rasterizer')
        self._driver = webdriver.Chrome(options=self._options)
        self._enable_download_headless()

    def _search_titles_libgen(self, book_name, metadata=None):
        """
        Search for book titles on Libgen based on the given book name.
        :param book_name: The name of the book to search for.
        :return: A DataFrame containing the search results.
        """
        tf = LibgenSearch()
        title_filters = [
            {"Extension": "epub", "Language": "English"}, 
            {"Extension": "mobi", "Language": "English"},
            {"Extension": "azw3", "Language": "English"}
        ]
        titles = []

        authors = metadata.get("Authors", [None]) if metadata else [None]
        book_name = metadata.get("Title", book_name) if metadata else book_name

        for filter in title_filters:
            for author in authors:
                search_term = f"{book_name} {author}" if author else book_name
                titles = tf.search_title_filtered(search_term, filter)
                if titles:
                    break
            if titles:
                break

        books = pd.DataFrame(titles)
        if not books.empty:
            books = books[["Title", "Mirror_1", "Mirror_2", "Mirror_3"]]

        return books
    
    def _log_not_found_book(self, book_name):
        """
        Log the book that was not found in a CSV file.
        :param book_name: The name of the book that was not found.
        """
        ic(f"Could not find the book '{book_name}'.")
        not_found_books = pd.DataFrame([book_name], columns=['Book Name'])

        # If the CSV file already exists, append the DataFrame to it
        if os.path.exists('Logs/not_found_books.csv'):
            not_found_books.to_csv('not_found_books.csv', mode='a', header=False, index=False)
        # If the CSV file does not exist, create it
        else:
            not_found_books.to_csv('Logs/not_found_books.csv', index=False)

    def _is_desired_book(self, book_name, book_title):
        """
        Uses GPT-3 to determine if the given book title matches the desired book name.

        :param book_name: The name of the book we're searching for.
        :param book_title: The title of a book to compare with the desired book name.
        :return: True if the book_title is the desired book, False otherwise.
        """
        prompt = f"I am searching for the book '{book_name}'. Is '{book_title}' the book I am looking for and is it in English?"
        completion = self._client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
             "content": """
             You are a highly knowledgeable assistant with expertise in books. Your task is to carefully compare two specific books. 
             When asked a question about these books, please respond only with 'yes' or 'no'. 
             It is crucial that you confirm whether each book is an original work or a summary. 
             If a book is identified as any form of summary, condensed version, or abridged edition, you must answer 'no'. 
             Otherwise, if it's an original, full-length work, answer 'yes'. Your accuracy in distinguishing between original works and summaries is essential.
             """
             },
            {"role": "user", "content": prompt}
        ])
        response = completion.choices[0].message.content
        return 'yes' in response.lower()
    
    def _backup_download(self, book_name, metadata=None):
        """
        Backup download method to use if the main download method fails.
        :param book_name: The name of the book to download.
        """
        ic(f"Searching for the '{book_name}' with Z-Library instead ...")

        authors = metadata.get("Authors", [None]) if metadata else [None]
        book_name = metadata.get("Title", book_name) if metadata else book_name

        for author in authors:
            search_term = f"{book_name} {author}" if author else book_name
            results = self._Z.search(message=search_term, languages=["English"], extensions="epub")["books"]
            books = pd.DataFrame(results)
            ic(books)

            book_to_download = next((row for _, row in books.iterrows() if self._is_desired_book(book_name, row["title"])), None)

            if book_to_download is not None:
                _, content = self._Z.downloadBook(book=book_to_download)
                with open(f"{self._books_dir}/{book_name}", "wb") as f:
                    f.write(content)
                ic(f"Successfully downloaded the book '{book_name}'.")
                break
        else:
            self._log_not_found_book(book_name)
            
        
        

    def _search_book(self, books, book_name):
        """
        Search for a specific book in the given DataFrame of books.
        :param books: The DataFrame containing the books.
        :param book_name: The name of the book to search for.
        :return: A DataFrame containing the details of the found book, or None if not found.
        """
        if books.empty:
            self._backup_download(book_name)
            return None

        for _, row in books.iterrows():
            book_title = row["Title"]
            if self._is_desired_book(book_name, book_title):
                ic(f"Found the book: '{book_name}'")
                return pd.DataFrame([row])

            ic(f"'{book_title}' is not the book we are searching for.")

        return None
    
    def _wait_for_download_complete(self, timeout=300, check_interval=10):
        """
        Waits for a file to download in the specified directory, using the .crdownload extension.
        :param timeout: Maximum time to wait for the download to complete.
        :param check_interval: Interval to check for download completion.
        :return: True if download is completed, False otherwise.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if there is a .crdownload file in the directory
            if any(file.endswith('.crdownload') for file in os.listdir(self._download_dir)):
                time.sleep(check_interval)
            else:
                # No .crdownload file found, download is complete
                return True

        return False  # Timed out
    
    def _resolve_download_links(self, link):
        """
        Resolve the download links from the given webpage link.
        :param link: The link to the webpage.
        :return: A dictionary containing the download links.
        """
 
        page = requests.get(link)
        soup = BeautifulSoup(page.text, "html.parser")
        links = soup.find_all("a", string=self.MIRROR_SOURCES)
        download_links = {link.string: link["href"] for link in links if link.string == "GET"}
        return download_links
    
    def _file_cleanup(self, book_name):
        """
        Perform cleanup operations on the downloaded file.
        :param book_name: The name of the book.
        """

        file_name = os.listdir(self._download_dir)[0]
        file_path = rename_file(os.path.join(self._download_dir, file_name), book_name)
        
        if not file_path.endswith('.epub'):
            if convert_to_epub(file_path, self._books_dir):
                ic(f"Successfully converted and moved {book_name}")
            else:
                move_file(file_path, self._books_dir)
        else:
            move_file(file_path, self._books_dir)

    def _check_empty_folder(self):
        """
        Perform cleanup operations on the downloaded file.
        :param book_name: The name of the book.
        """

        return not bool(os.listdir(self._download_dir))
    
    def _process_download_link(self, download_link, book_name):
        """
        Processes a download link to download a book and clean up the files.

        This method resolves the download link, initializes a web driver, navigates to the 
        download link, waits for the download to complete, and then cleans up the files.

        :param download_link: The link to download the book.
        :param book_name: The name of the book to be downloaded.
        """
        link = self._resolve_download_links(download_link)
        self._initialize_driver()
        self._driver.get(link['GET'])
        time.sleep(4)
        self._wait_for_download_complete()
        self._driver.quit()

        self._file_cleanup(book_name)
    
    def _process_mirror_links(self, row, mirror_list):
        """
        Processes a list of mirror links to download a book.

        This method iterates over the mirror links, resolves each download link, initializes a 
        web driver, navigates to the download link, waits for the download to complete, and 
        checks if the download was successful. If a download was successful, it quits the driver 
        and returns True. If none of the downloads were successful, it returns False.

        :param row: The row in the DataFrame containing the download links.
        :param mirror_list: The list of mirror links to process.
        :return: True if a download was successful, False otherwise.
        """
        for mirror in mirror_list:
            link = self._resolve_download_links(row[mirror])
            self._driver.get(link['GET'])
            time.sleep(4)
            self._wait_for_download_complete()
            if link and not self._check_empty_folder():
                self._driver.quit()
                return True
        return False

    def _process_download_links(self, download_links, book_name):
        """
        Processes a DataFrame of download links to download a book.

        This method initializes a web driver, iterates over the download links, and processes 
        each link using a list of mirror links. If a download is successful, it breaks the loop 
        and cleans up the files. If no download is successful, it still cleans up the files.

        :param download_links: A DataFrame containing the download links.
        :param book_name: The name of the book to be downloaded.
        """
        self._initialize_driver()
        for _, row in download_links.iterrows():
            if self._process_mirror_links(row, self.MIRROR_LIST):
                break

        self._file_cleanup(book_name)

    def _auto_download_book(self, book_name, download_links=None, download_link=None):
        """
        Automatically download the book using the provided download links or download link.
        :param book_name: The name of the book.
        :param download_links: A DataFrame containing the download links.
        :param download_link: The direct download link.
        """
        if download_link:
            self._process_download_link(download_link, book_name)
        elif not download_links.empty:
            self._process_download_links(download_links, book_name)
                        
        else:
            self._log_not_found_book(book_name)
    
    def _download_book_manually(self, book_name, download_link):
        """
        Download the book manually using the provided download link.
        :param book_name: The name of the book.
        :param download_link: The direct download link.
        """

        self._initialize_driver()
        ic(f"Downloading the book '{book_name}'...")
        self._driver.get(download_link)
        time.sleep(10)

        download_link = self._driver.find_element(By.XPATH, '//*[@id="main"]/tbody/tr[1]/td[2]/a')
        download_link.click()
        time.sleep(10)
        self._wait_for_download_complete()
        self._driver.quit()

        self._file_cleanup(book_name)

    def scrape_book(self, book_name, download_link=None, download_links=None):
        """
        Scrape and download the book based on the provided book name and download link(s).
        :param book_name: The name of the book.
        :param download_link: The direct download link.
        :param download_links: A DataFrame containing the download links.
        """

        
        if download_link:
            self._auto_download_book(book_name=book_name, download_link=download_link)
        elif download_links is not None and not download_links.empty:
            self._auto_download_book(book_name=book_name, download_links=download_links)
        else:
            ic(f"Searching for the book '{book_name}'...")
            metadata = meta(isbn_from_words(book_name))
            book_name = metadata.get("Title", book_name) if metadata else book_name
            books = self._search_titles_libgen(book_name, metadata=metadata)
            book = self._search_book(books, book_name)

            if book is not None:
                book_name = book['Title'].values[0]
                links = book[['Mirror_1', 'Mirror_2', 'Mirror_3']]
                self._auto_download_book(book_name=book_name, download_links=links)


        
